"""Gradio app for MOS (Mean Opinion Score) evaluation of TTS audio samples."""

import os
import random
import string
from datetime import datetime

import gradio as gr
import pandas as pd


def generate_random_id(length=8):
    """Generate a random anonymous ID."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


# JavaScript for cookie handling
COOKIE_JS = """
function() {
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
        return null;
    }

    function generateId(length = 8) {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
        let result = '';
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }

    let existingId = getCookie('mos_listener_id');
    if (!existingId) {
        existingId = generateId();
        // Save new ID to cookie (expires in 10 days)
        const expires = new Date(Date.now() + 10 * 24 * 60 * 60 * 1000).toUTCString();
        document.cookie = `mos_listener_id=${existingId}; expires=${expires}; path=/; SameSite=Strict`;
    }
    return existingId;
}
"""

SET_COOKIE_JS = """
function(listener_id) {
    // Set cookie to expire in 10 days
    const expires = new Date(Date.now() + 10 * 24 * 60 * 60 * 1000).toUTCString();
    document.cookie = `mos_listener_id=${listener_id}; expires=${expires}; path=/; SameSite=Strict`;
    return listener_id;
}
"""


# --- CONFIG ---
AUDIO_DIR = "evaluation_samples/audio"
TEXTS_CSV = "evaluation_samples/texts.csv"
RESULTS_FILE = "evaluation_samples/mos_results.csv"

# Load text metadata
texts_df = pd.read_csv(TEXTS_CSV, sep=";")
texts_map = {row["filename"]: row["text"] for _, row in texts_df.iterrows()}

# Collect all audio files with model info
audio_files = []
for model in os.listdir(AUDIO_DIR):
    model_path = os.path.join(AUDIO_DIR, model)
    if os.path.isdir(model_path):
        for f in os.listdir(model_path):
            if f.endswith(".wav"):
                audio_files.append(
                    {
                        "filepath": os.path.join(model_path, f),
                        "filename": f,
                        "model": model,
                        "text": texts_map.get(f, ""),
                    }
                )


# --- UTILITY FUNCTIONS ---
def get_shuffled_files(listener_id):
    """Get a deterministic shuffle of files based on listener ID."""
    files = audio_files.copy()
    # Use listener_id as seed for deterministic order
    random.seed(listener_id)
    random.shuffle(files)
    random.seed()  # Reset to random
    return files


def get_seen_files(listener_id):
    """Get set of files already rated by this listener."""
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        listener_df = df[df["listener"] == listener_id]
        if "model" in listener_df.columns and "filename" in listener_df.columns:
            return set(
                (
                    listener_df["model"].astype(str)
                    + "/"
                    + listener_df["filename"].astype(str)
                ).tolist()
            )
        return set(listener_df["filename"].tolist())
    return set()


def start_session(existing_id):
    """Initialise a listener session and return the first audio sample to rate."""
    # Use existing ID from cookie if available, otherwise generate new one
    listener_id = existing_id if existing_id else generate_random_id()

    seen_files = get_seen_files(listener_id)
    all_files = get_shuffled_files(listener_id)

    # Find current index (how many we've completed)
    index = len(seen_files)

    # Get remaining files (those not yet seen, in deterministic order)
    remaining = [
        f for f in all_files if f"{f['model']}/{f['filename']}" not in seen_files
    ]

    if not remaining:
        return (
            None,
            "",
            "Alle samples er vurderet. Tak for hjælpen!",
            gr.update(value=None),
            "",
            listener_id,
            index,
        )

    item = remaining[0]
    total = len(all_files)
    progress = f"Sample {index + 1} / {total}"

    return (
        item["filepath"],
        item["text"],
        progress,
        gr.update(value=None),
        "",
        listener_id,
        index,
    )


def submit_rating(rating, listener_id, current_index):
    """Save a rating for the current sample and advance to the next one."""
    if not rating or not listener_id:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
            listener_id,
            current_index,
        )

    seen_files = get_seen_files(listener_id)
    all_files = get_shuffled_files(listener_id)
    remaining = [
        f for f in all_files if f"{f['model']}/{f['filename']}" not in seen_files
    ]

    if not remaining:
        return (
            None,
            "",
            "Færdig.",
            gr.update(value=None),
            "",
            listener_id,
            current_index,
        )

    item = remaining[0]

    # Extract numeric score (e.g., "3.25" from "3.25 - Fair")
    numeric_rating = rating.split(" - ")[0]

    row = {
        "time": datetime.now(),
        "listener": listener_id,
        "filename": item["filename"],
        "model": item["model"],  # hidden from evaluator
        "rating": numeric_rating,
    }

    df = pd.DataFrame([row])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)

    # Get next item
    new_index = current_index + 1
    new_seen = seen_files | {f"{item['model']}/{item['filename']}"}
    new_remaining = [
        f for f in all_files if f"{f['model']}/{f['filename']}" not in new_seen
    ]

    if not new_remaining:
        return (
            None,
            "",
            "Alle samples er vurderet. Tak for hjælpen!",
            gr.update(value=None),
            "",
            listener_id,
            new_index,
        )

    next_item = new_remaining[0]
    total = len(all_files)
    progress = f"Sample {new_index + 1} / {total}"

    return (
        next_item["filepath"],
        next_item["text"],
        progress,
        gr.update(value=None),
        "",
        listener_id,
        new_index,
    )


# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("# MOS-evaluering af TTS-modeller")

    # Hidden state components
    listener_id = gr.Textbox(visible=False, value="")
    current_index = gr.State(0)

    audio_player = gr.Audio(label="Lyt", type="filepath")
    reference_text = gr.Textbox(label="Referencetekst", interactive=False)
    progress = gr.Markdown()
    rating = gr.Dropdown(
        choices=[
            "1.00 - Dårlig (Talen lyder helt unaturlig og kunstig. Den syntetiske stemme er meget tydelig og så generende, at den kan være svær at lytte til.)",
            "1.50 - Dårlig",
            "2.00 - Ringe (Talen lyder for det meste unaturlig. Den syntetiske kvalitet er tydelig og opleves som generende, men den er stadig til at holde ud.)",
            "2.50 - Ringe",
            "3.00 - Middel (Talen fremstår både naturlig og unaturlig i nogenlunde lige grad. Det er tydeligt, at stemmen er syntetisk, og det kan virke en smule forstyrrende.)",
            "3.50 - Middel",
            "4.00 - God (Talen lyder overvejende naturlig. Man kan godt høre, at det er syntetisk tale, men det er kun svagt mærkbart og ikke generende.)",
            "4.50 - God",
            "5.00 - Fremragende (Talen lyder helt naturlig og kan ikke skelnes fra en rigtig menneskestemme. Der er ingen hørbare tegn på, at det er syntetisk tale.)",
        ],
        label="Vurdér kvalitet",
        value="3.00 - Middel (Talen fremstår både naturlig og unaturlig i nogenlunde lige grad. Det er tydeligt, at stemmen er syntetisk, og det kan virke en smule forstyrrende.)",
    )
    error_msg = gr.Markdown()
    submit_btn = gr.Button("Indsend og fortsæt")

    # Load cookie and start session on page load
    demo.load(
        fn=start_session,
        inputs=[listener_id],
        outputs=[
            audio_player,
            reference_text,
            progress,
            rating,
            error_msg,
            listener_id,
            current_index,
        ],
        js=COOKIE_JS,
    ).then(fn=None, inputs=[listener_id], outputs=[listener_id], js=SET_COOKIE_JS)

    submit_btn.click(
        submit_rating,
        inputs=[rating, listener_id, current_index],
        outputs=[
            audio_player,
            reference_text,
            progress,
            rating,
            error_msg,
            listener_id,
            current_index,
        ],
    )

demo.launch(server_name="0.0.0.0")
