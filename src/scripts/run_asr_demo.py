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
from omegaconf import DictConfig
from punctfix import PunctFixer
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("asr-demo")
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(config_path="../../config", config_name="demo_hoert", version_base=None)
def main(config: DictConfig) -> None:
    """Run the ASR demo.

    Args:
        config:
            The Hydra configuration for the demo.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading the ASR model...")
    model_id = config.model_id

    transcriber = pipeline(
        task="automatic-speech-recognition", model=model_id, device=device
    )

    logger.info("Loading the punctuation fixer model...")
    transcription_fixer = PunctFixer(language="da", device=device)

    logger.info("Models loaded, ready to transcribe audio.")

    # Function to load and preprocess audio
    def load_audio(file_path, target_sample_rate=16_000):
        """Load and preprocess audio from a file. resample to 16kHz if necessary, and convert to mono."""
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if necessary
        if sample_rate != target_sample_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            )
            waveform = transform(waveform)

        return waveform.squeeze(0).numpy()

    # Function to transcribe audio
    def transcribe(audio_file):
        """Transcribe an audio file."""
        if not audio_file:
            return (
                "No audio was provided. Please record or upload an audio clip, and try "
                "again."
            )

        audio = load_audio(audio_file, config.sampling_rate)

        logger.info(f"Transcribing audio clip of {len(audio) / 16_000:.2f} seconds...")
        if config.model_type == "whisper":
            forced_decoder_ids = transcriber.tokenizer.get_decoder_prompt_ids(
                language="danish", task="transcribe"
            )

            transcription = transcriber(
                inputs=audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
            )
        else:
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

    # Gradio Interface with both microphone and file upload

    image_path = "resources/CoRal.png"

    html_content = """
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <a href="https://www.alvenir.ai/">
                <img src="https://static.wixstatic.com/media/b5799d_f473a732bdce46ca91c6f02f963309bc~mv2.png/v1/fill/w_164,h_88,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/b5799d_f473a732bdce46ca91c6f02f963309bc~mv2.png" alt="Partner 2" style="height: 100px; margin: 10px;">
            </a>
            <a href="https://www.corti.ai/">
                <img src="https://cdn.prod.website-files.com/679910de24e675a93f045f3b/679917a13caaa280dd45d39b_corti-logo.svg" alt="Partner 1" style="height: 100px; margin: 10px;">
            </a>
            <a href="https://di.ku.dk/">
                <img src="https://designguide.ku.dk/download/co-branding/ku_logo_dk_h.png" alt="Partner 2" style="height: 100px; margin: 10px;">
            </a>
            <a href="https://digst.dk/">
                <img src="https://digst.dk/media/cdld0gfa/digst_logo_dk_aubergine_rgb_new_png.png?format=webp&quality=92" alt="Partner 2" style="height: 100px; margin: 10px;">
            </a>
            <a href="https://alexandra.dk/">
                <img src="https://alexandra.dk/wp-content/uploads/2020/02/Alexandra-Instituttet_Logo_Sort_DK.webp" alt="Partner 2" style="height: 100px; margin: 10px;">
            </a>
        </div>
    """

    demo = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
        outputs="text",
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
    )

    # Create a Gradio Blocks layout to include the image at the bottom
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as app:
        gr.Image(
            image_path,
            interactive=False,
            container=False,
            show_share_button=False,
            show_download_button=False,
            show_fullscreen_button=False,
            show_label=False,
        )

        demo.render()  # Render the Interface

        gr.Markdown("")
        gr.Markdown("")
        gr.Markdown("")
        gr.Markdown(
            """
            # Projekt Partnere:
            """
        )
        gr.HTML(html_content)

    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
