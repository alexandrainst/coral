"""Run an ASR demo using Gradio.

Usage:
    python src/scripts/run_asr_demo.py [key=value] [key=value] ...
"""

import logging
import warnings

import gradio as gr
import hydra
import numpy as np
import samplerate
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from punctfix import PunctFixer
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("roest-asr-demo")

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()


@hydra.main(config_path="../../config", config_name="demo", version_base=None)
def main(config: DictConfig) -> None:
    """Run the ASR demo.

    Args:
        config:
            The Hydra configuration for the demo.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info("Loading the ASR model...")
    transcriber = pipeline(
        task="automatic-speech-recognition", model=config.model_id, device=device
    )

    logger.info("Loading the punctuation fixer model...")
    transcription_fixer = PunctFixer(language="da", device=device)

    logger.info("Models loaded, ready to transcribe audio.")

    def transcribe_audio(sampling_rate_and_audio: tuple[int, np.ndarray] | None) -> str:
        """Transcribe the audio.

        Args:
            sampling_rate_and_audio:
                A tuple with the sampling rate and the audio, or None if no audio is
                provided.

        Returns:
            The transcription.
        """
        if sampling_rate_and_audio is None:
            return (
                "No audio was provided. Please record or upload an audio clip, and try "
                "again."
            )

        sampling_rate, audio = sampling_rate_and_audio
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = samplerate.resample(
            audio, config.sampling_rate / sampling_rate, "sinc_best"
        )

        logger.info(f"Transcribing audio clip of {len(audio) / 16_000:.2f} seconds...")
        transcription = transcriber(
            inputs=audio, generate_kwargs=dict(language="danish", task="transcribe")
        )
        if not isinstance(transcription, dict):
            return ""

        logger.info(
            f"Raw transcription is {transcription['text']!r}. Cleaning it up..."
        )
        cleaned_transcription = transcription_fixer.punctuate(
            text=transcription["text"]
        )

        logger.info(f"Final transcription: {cleaned_transcription!r}")
        return cleaned_transcription

    demo = gr.Interface(
        fn=transcribe_audio,
        inputs=gr.Audio(sources=["microphone", "upload"]),
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
