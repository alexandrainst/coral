"""Finetune a Qwen3-TTS speech synthesis model on the CoRal-TTS dataset."""

import importlib.util
import json
import os
import shutil
import typing as t
from functools import partial

import librosa
import numpy as np
import torch
from accelerate import Accelerator
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

if importlib.util.find_spec("qwen_tts") is not None:
    from qwen_tts import Qwen3TTSTokenizer
    from qwen_tts.core.models import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from qwen_tts.inference.qwen3_tts_model import AudioLike, Qwen3TTSModel


TTS_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

T = t.TypeVar("T")


class TTSDataset(torch.utils.data.Dataset):
    """A dataset for training the TTS model."""

    def __init__(
        self,
        dataset: Dataset,
        processor: PreTrainedTokenizer,
        config: Qwen3TTSConfig,
        lag_num: int = -1,
    ) -> None:
        """Initialise the dataset.

        Args:
            dataset:
                The raw dataset.
            processor:
                The processor to use.
            config:
                The config to use.
            lag_num:
                The number of lags to use.
        """
        self.dataset = dataset
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get the item at the given index.

        Args:
            index:
                The index to get the item from.

        Returns:
            The item at the given index.
        """
        item = self.dataset[index]

        # Process the text
        text = (
            f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        )
        text_ids = self._tokenize_texts(text=text)

        input = self.processor(text=text, return_tensors="pt", padding=True)
        text_ids = (
            input["input_ids"].unsqueeze(0)
            if input["input_ids"].dim() == 1
            else input["input_ids"]
        )

        # Get the Mel spectrogram
        ref_audio_list = (
            item["ref_audio"]
            if isinstance(item["ref_audio"], list)
            else [item["ref_audio"]]
        )
        audio, sr = normalise_audio_inputs(audios=ref_audio_list)[0]
        ref_mel = extract_mels(audio=audio, sr=sr)

        return dict(
            text_ids=text_ids[:, :-5],
            audio_codes=torch.tensor(item["audio_codes"], dtype=torch.long),
            ref_mel=ref_mel,
        )


def prepare_data(dataset_id: str, speaker: t.Literal["mic", "nic"]) -> Dataset:
    """Prepare the data for training.

    Args:
        dataset_id:
            The dataset ID.
        speaker:
            The speaker to use.

    Returns:
        The prepared dataset.
    """
    # Load the dataset
    dataset = load_dataset(path=dataset_id, split="train")
    assert isinstance(dataset, Dataset), (f"Expected a Dataset, got {type(dataset)}",)

    # Filter the dataset
    dataset = dataset.filter(lambda x: x["speaker_id"] == speaker)
    dataset = dataset.filter(lambda x: x["audio"]["array"].shape[0] > 0)

    # Tokenise the audio
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="auto"
    )
    dataset = dataset.map(
        lambda batch: dict(
            audio_codes=tokenizer.encode(
                audios=[x["array"] for x in batch["audio"]],
                sr=batch["audio"][0]["sampling_rate"],
            ).audio_codes
        ),
        batched=True,
        batch_size=32,
    )
    assert isinstance(dataset, Dataset), (
        f"Expected a Dataset after tokenisation, got {type(dataset)}",
    )

    return dataset


def train_tts_model(dataset: Dataset) -> None:
    """Train the TTS model.

    Args:
        dataset:
            The dataset to train on.
    """
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")

    model = Qwen3TTSModel.from_pretrained(
        TTS_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = Qwen3TTSConfig.from_pretrained(TTS_BASE_MODEL)
    tts_dataset = TTSDataset(dataset=dataset, processor=model.tokenizer, config=config)

    dataloader = DataLoader(
        tts_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            tts_bos_token_id=config.tts_bos_token_id,
            tts_eos_token_id=config.tts_eos_token_id,
            tts_pad_token_id=config.tts_pad_token_id,
            codec_bos_token_id=config.talker_config.codec_bos_id,
            codec_eos_token_id=config.talker_config.codec_eos_token_id,
            codec_pad_token_id=config.talker_config.codec_pad_id,
            codec_nothink_id=config.talker_config.codec_nothink_id,
            codec_think_bos_id=config.talker_config.codec_think_bos_id,
            codec_think_eos_id=config.talker_config.codec_think_eos_id,
        ),
    )

    optimizer = AdamW(model.model.parameters(), lr=2e-5, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        model.model, optimizer, dataloader
    )

    model.train()

    for epoch in range(3):
        speaker_embedding = None
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids)
                input_codec_embedding = model.talker.model.codec_embedding(
                    input_codec_ids
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = (
                        model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = (
                    model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )
                )

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        if accelerator.is_main_process and speaker_embedding is not None:
            output_dir = os.path.join("output", f"checkpoint-epoch-{epoch}")
            shutil.copytree(TTS_BASE_MODEL, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(TTS_BASE_MODEL, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {"speaker_test": 3000}
            talker_config["spk_is_dialect"] = {"speaker_test": False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()
            }

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][3000] = (
                speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)


@torch.inference_mode()
def extract_mels(audio: np.ndarray, sr: int) -> torch.Tensor:
    """Extract the mel spectrograms.

    Args:
        audio:
            The audio to extract the mels from.
        sr:
            The sampling rate of the audio.

    Returns:
        The mel spectrograms.
    """
    assert sr == 24000, "Only 24kHz audio is supported"
    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    return mels


def normalise_audio_inputs(
    audios: AudioLike | list[AudioLike],
) -> list[tuple[np.ndarray, int]]:
    """Normalise audio inputs into a list of (waveform, sr).

    Supported forms:
      - str: wav path / URL / base64 audio string
      - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
      - (np.ndarray, sr): waveform + sampling rate
      - list of the above

    Args:
        audios:
            Audio input(s).

    Returns:
        List of (float32 waveform, original sr).

    Raises:
        ValueError:
            If a numpy waveform is provided without sr.
        TypeError:
            If the input type is not supported.
    """
    if isinstance(audios, list):
        items = audios
    else:
        items = [audios]

    out: list[tuple[np.ndarray, int]] = []
    for a in items:
        if isinstance(a, str):
            audio, sr = librosa.load(path=a, sr=None, mono=True)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            audio = audio.astype(np.float32)
            sr = int(sr)
            out.append((audio, sr))
        elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
            out.append((a[0].astype(np.float32), int(a[1])))
        elif isinstance(a, np.ndarray):
            raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
        else:
            raise TypeError(f"Unsupported audio input type: {type(a)}")
    return out


def collate_fn(
    batch: list[dict],
    tts_bos_token_id: int,
    tts_eos_token_id: int,
    tts_pad_token_id: int,
    codec_bos_token_id: int,
    codec_eos_token_id: int,
    codec_pad_token_id: int,
    codec_nothink_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
) -> dict:
    """Collate function for the dataloader.

    Args:
        batch:
            The batch to collate.
        tts_bos_token_id:
            The beginning of sequence token ID for the TTS model.
        tts_eos_token_id:
            The end of sequence token ID for the TTS model.
        tts_pad_token_id:
            The padding token ID for the TTS model.
        codec_bos_token_id:
            The beginning of sequence token ID for the codec.
        codec_eos_token_id:
            The end of sequence token ID for the codec.
        codec_pad_token_id:
            The padding token ID for the codec.
        codec_nothink_id:
            The nothink token ID for the codec.
        codec_think_bos_id:
            The beginning of sequence token ID for the codec.
        codec_think_eos_id:
            The end of sequence token ID for the codec.

    Returns:
        The collated batch.
    """
    item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
    max_length = max(item_length) + 8
    b, t = len(batch), max_length

    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

    for i, data in enumerate(batch):
        text_ids = data["text_ids"]
        audio_codec_0 = data["audio_codes"][:, 0]
        audio_codecs = data["audio_codes"]

        text_ids_len = text_ids.shape[1]
        codec_ids_len = audio_codec_0.shape[0]

        # text channel
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:7, 0] = tts_pad_token_id
        input_ids[i, 7, 0] = tts_bos_token_id
        input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
        input_ids[i, 8 + text_ids_len - 3, 0] = tts_eos_token_id
        input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = (
            tts_pad_token_id
        )
        text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        # codec channel
        input_ids[i, 3:8, 1] = torch.tensor(
            [
                codec_nothink_id,
                codec_think_bos_id,
                codec_think_eos_id,
                0,  # for speaker embedding
                codec_pad_token_id,
            ]
        )
        input_ids[i, 8 : 8 + text_ids_len - 3, 1] = codec_pad_token_id
        input_ids[i, 8 + text_ids_len - 3, 1] = codec_pad_token_id
        input_ids[i, 8 + text_ids_len - 2, 1] = codec_bos_token_id
        input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = (
            audio_codec_0
        )
        input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = codec_eos_token_id

        codec_0_labels[
            i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len
        ] = audio_codec_0
        codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = codec_eos_token_id

        codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = (
            audio_codecs
        )

        codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
        codec_embedding_mask[i, 6] = False  # for speaker embedding

        codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = (
            True
        )
        attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

    ref_mels = [data["ref_mel"] for data in batch]
    ref_mels = torch.cat(ref_mels, dim=0)

    return dict(
        input_ids=input_ids,
        ref_mels=ref_mels,
        attention_mask=attention_mask,
        text_embedding_mask=text_embedding_mask.unsqueeze(-1),
        codec_embedding_mask=codec_embedding_mask.unsqueeze(-1),
        codec_0_labels=codec_0_labels,
        codec_ids=codec_ids,
        codec_mask=codec_mask,
    )
