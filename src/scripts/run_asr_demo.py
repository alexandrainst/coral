"""Run an ASR demo using Gradio.

Usage:
    python src/scripts/run_asr_demo.py [key=value] [key=value] ...
"""

import warnings

import gradio as gr
import hydra
import numpy as np
import samplerate
import torch
from omegaconf import DictConfig
from punctfix import PunctFixer
from transformers import pipeline

warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(config_path="../../config", config_name="demo", version_base=None)
def main(config: DictConfig) -> None:
    """Run the ASR demo.

    Args:
        config:
            The Hydra configuration for the demo.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transcriber = pipeline(
        task="automatic-speech-recognition", model=config.model_id, device=device
    )
    transcription_fixer = PunctFixer(language="da", device=device)

    def transcribe_audio(sampling_rate_and_audio: tuple[int, np.ndarray]) -> str:
        """Transcribe the audio.

        Args:
            sampling_rate_and_audio:
                A tuple with the sampling rate and the audio.

        Returns:
            The transcription.
        """
        sampling_rate, audio = sampling_rate_and_audio
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = samplerate.resample(
            audio, config.sampling_rate / sampling_rate, "sinc_best"
        )
        transcription = transcriber(inputs=audio)
        if not isinstance(transcription, dict):
            return ""
        cleaned_transcription = transcription_fixer.punctuate(
            text=transcription["text"]
        )
        return cleaned_transcription

    demo = gr.Interface(
        fn=transcribe_audio,
        inputs=gr.Audio(sources=["microphone", "upload"]),
        outputs="textbox",
        title=config.title,
        description=config.description,
        allow_flagging="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
