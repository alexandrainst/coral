"""Run an ASR demo using Gradio.

Usage:
    python src/scripts/run_asr_demo.py [key=value] [key=value] ...
"""

import logging
import warnings

import gradio as gr
import hydra
import torch
import torchaudio
from dotenv import load_dotenv
from omegaconf import DictConfig
from punctfix import PunctFixer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hoert-asr-demo")

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()


@hydra.main(config_path="../../config", config_name="demo_hoert", version_base=None)
def main(config: DictConfig) -> None:
    """Run the ASR demo.

    Args:
        config:
            The Hydra configuration for the demo.
    """
    logger.info("Loading the ASR model...")

    model_id = config.model_id
    # model_id = "openai/whisper-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)
    model.generation_config = generation_config

    logger.info("Loading the punctuation fixer model...")
    transcription_fixer = PunctFixer(language="da", device=device)

    logger.info("Models loaded, ready to transcribe audio.")

    # Function to load and preprocess audio
    def load_audio(file_path):
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if necessary
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = transform(waveform)

        return waveform.squeeze(0), 16000

    # Function to transcribe audio
    def transcribe(audio_file):
        if not audio_file:
            return "No audio provided."

        audio, sample_rate = load_audio(audio_file)
        logger.info(f"Transcribing audio clip of {len(audio) / 16_000:.2f} seconds...")

        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            predicted_ids = model.generate(**inputs)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ]

        logger.info(f"Raw transcription is {transcription!r}. Cleaning it up...")
        cleaned_transcription = transcription_fixer.punctuate(text=transcription)

        logger.info(f"Final transcription: {cleaned_transcription!r}")
        return cleaned_transcription

    demo = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
        outputs="textbox",
        title=config.title,
        description=config.description,
        css="p { font-size: 1.0rem; }",
        allow_flagging="never",
        examples=[
            "https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/audio-examples/bornholmsk.wav",
            "https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/audio-examples/soenderjysk.wav",
            "https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/audio-examples/nordjysk.wav",
            "https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/audio-examples/accent.wav",
        ],
        cache_examples=False,
        theme=gr.themes.Soft(primary_hue="orange"),
    )
    demo.launch()


if __name__ == "__main__":
    main()
