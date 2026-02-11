"""
Multi-speaker text-to-speech inference script for FastSpeech2.
Run from project root. Supports speaker selection and reports generation time.
"""
import os
import re
import sys
import json
import argparse
import time
from string import punctuation

# Run from project root: add FastSpeech2 for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FASTSPEECH2_DIR = os.path.join(PROJECT_ROOT, "FastSpeech2")
if FASTSPEECH2_DIR not in sys.path:
    sys.path.insert(0, FASTSPEECH2_DIR)

os.chdir(FASTSPEECH2_DIR)  # so relative paths in configs (ckpt, preprocess) resolve

import torch
import yaml
import numpy as np
from scipy.io import wavfile

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    from g2p_en import G2p
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)


def load_speakers(preprocess_config):
    path = os.path.join(
        preprocess_config["path"]["preprocessed_path"],
        "speakers.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer(
    model,
    vocoder,
    configs,
    text,
    speaker_id,
    pitch_control=1.0,
    energy_control=1.0,
    duration_control=1.0,
    language="en",
):
    preprocess_config, model_config, train_config = configs
    if language == "en":
        texts = np.array([preprocess_english(text, preprocess_config)])
    else:
        raise NotImplementedError("Only English is supported in this script.")
    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_id], dtype=np.int64)
    text_lens = np.array([len(texts[0])])
    max_src_len = max(text_lens)
    batch = (ids, raw_texts, speakers, texts, text_lens, max_src_len)
    batch = to_device(batch, device)

    with torch.no_grad():
        t_start = time.perf_counter()
        output = model(
            *(batch[2:]),
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control,
        )
        t_model = time.perf_counter() - t_start

        mel_predictions = output[1].transpose(1, 2)
        lengths = output[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
        t_voc_start = time.perf_counter()
        wav_predictions = vocoder_infer(
            mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
        )
        t_vocoder = time.perf_counter() - t_voc_start

    total_time = time.perf_counter() - t_start
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wav = wav_predictions[0]
    duration_sec = len(wav) / sampling_rate
    rtf = total_time / duration_sec if duration_sec > 0 else float("inf")
    mel_len = output[9][0].item()
    mel_pred = output[1][0, :mel_len].detach().cpu().numpy().T  # (n_mel, time)
    duration = output[5][0, : batch[4][0].item()].detach().cpu().numpy()
    pitch = output[2][0, : batch[4][0].item()].detach().cpu().numpy()
    energy = output[3][0, : batch[4][0].item()].detach().cpu().numpy()

    return {
        "wav": wav,
        "sampling_rate": sampling_rate,
        "time_model_sec": t_model,
        "time_vocoder_sec": t_vocoder,
        "time_total_sec": total_time,
        "audio_duration_sec": duration_sec,
        "rtf": rtf,
        "mel": mel_pred,
        "pitch": pitch,
        "energy": energy,
        "duration": duration,
        "stats_path": preprocess_config["path"]["preprocessed_path"],
    }


def run_interactive():
    """Prompt for all options in the terminal and run inference."""
    print("=== FastSpeech2 Multi-Speaker TTS (interactive) ===\n")
    text = input("Enter text to synthesize [Hello world]: ").strip() or "Hello world"
    n_speakers = len(load_speakers(
        yaml.load(open("config/LibriTTS/preprocess.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    ))
    while True:
        try:
            speaker_id = input(f"Speaker ID (0-{n_speakers - 1}) [0]: ").strip() or "0"
            speaker_id = int(speaker_id)
            if 0 <= speaker_id < n_speakers:
                break
        except ValueError:
            pass
        print(f"Enter a number between 0 and {n_speakers - 1}.")
    restore_step = input("Checkpoint step [800000]: ").strip() or "800000"
    restore_step = int(restore_step)
    save_fig = input("Save mel spectrogram PNG? (y/n) [n]: ").strip().lower() == "y"
    pitch = input("Pitch control (1.0 = normal) [1.0]: ").strip() or "1.0"
    energy = input("Energy control (1.0 = normal) [1.0]: ").strip() or "1.0"
    duration = input("Duration/speed control (1.0 = normal) [1.0]: ").strip() or "1.0"
    pitch_control = float(pitch)
    energy_control = float(energy)
    duration_control = float(duration)

    # Load configs
    preprocess_config = yaml.load(
        open("config/LibriTTS/preprocess.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("config/LibriTTS/model.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("config/LibriTTS/train.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)
    class Args:
        restore_step = restore_step
    model = get_model(Args(), configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    language = preprocess_config["preprocessing"]["text"].get("language", "en")
    result = infer(
        model, vocoder, configs, text, speaker_id,
        pitch_control=pitch_control, energy_control=energy_control,
        duration_control=duration_control, language=language,
    )
    basename = re.sub(r"[^\w\s-]", "", text)[:50].strip() or "output"
    basename = re.sub(r"[-\s]+", "_", basename)
    output_dir = "./output/result/LibriTTS"
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, f"{basename}.wav")
    wavfile.write(wav_path, result["sampling_rate"], result["wav"])
    print(f"\nSaved: {wav_path}")
    if save_fig and result.get("mel") is not None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from utils.tools import plot_mel, expand
        with open(os.path.join(result["stats_path"], "stats.json"), "r") as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]
        pitch_arr, energy_arr, duration_arr = result["pitch"], result["energy"], result["duration"]
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch_arr = expand(pitch_arr, duration_arr)
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy_arr = expand(energy_arr, duration_arr)
        mel_len = result["mel"].shape[1]
        pitch_arr = pitch_arr[:mel_len] if len(pitch_arr) > mel_len else np.pad(pitch_arr, (0, mel_len - len(pitch_arr)))
        energy_arr = energy_arr[:mel_len] if len(energy_arr) > mel_len else np.pad(energy_arr, (0, mel_len - len(energy_arr)))
        fig = plot_mel([(result["mel"], pitch_arr, energy_arr)], stats, ["Synthesized Spectrogram"])
        fig_path = os.path.join(output_dir, f"{basename}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved figure: {fig_path}")
    print("\n--- Generation time ---")
    print(f"  Speaker ID:        {speaker_id} (of {n_speakers} speakers)")
    print(f"  Model forward:     {result['time_model_sec']:.4f} s")
    print(f"  Vocoder:           {result['time_vocoder_sec']:.4f} s")
    print(f"  Total:             {result['time_total_sec']:.4f} s")
    print(f"  Audio duration:    {result['audio_duration_sec']:.4f} s")
    print(f"  RTF:               {result['rtf']:.4f}")


def main():
    if len(sys.argv) == 1:
        run_interactive()
        return
    if "--gui" in sys.argv or "-gui" in sys.argv:
        launch_gui()
        return

    parser = argparse.ArgumentParser(
        description="Multi-speaker FastSpeech2 TTS inference with generation time and speaker selection."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello world",
        help="Input text to synthesize.",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker synthesis (0 to n_speakers-1). Use --list_speakers to see IDs.",
    )
    parser.add_argument(
        "--list_speakers",
        action="store_true",
        help="List available speaker IDs (from speakers.json) and exit.",
    )
    parser.add_argument(
        "--restore_step",
        type=int,
        default=800000,
        help="Checkpoint step to load (e.g. 800000).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/result/LibriTTS",
        help="Directory to save output WAV (and optional PNG).",
    )
    parser.add_argument(
        "--save_fig",
        action="store_true",
        help="Save mel spectrogram figure as PNG.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Base name for output files (default: derived from text, sanitized).",
    )
    parser.add_argument(
        "-p", "--preprocess_config",
        type=str,
        default="config/LibriTTS/preprocess.yaml",
        help="Path to preprocess.yaml (relative to FastSpeech2).",
    )
    parser.add_argument(
        "-m", "--model_config",
        type=str,
        default="config/LibriTTS/model.yaml",
        help="Path to model.yaml (relative to FastSpeech2).",
    )
    parser.add_argument(
        "-t", "--train_config",
        type=str,
        default="config/LibriTTS/train.yaml",
        help="Path to train.yaml (relative to FastSpeech2).",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="Pitch control (larger = higher pitch).",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="Energy control (larger = louder).",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="Duration/speed control (larger = slower).",
    )
    args = parser.parse_args()

    # Load configs (paths relative to FastSpeech2 after chdir)
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(args.model_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(args.train_config, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)

    speakers_dict = load_speakers(preprocess_config)
    n_speakers = len(speakers_dict)
    speaker_ids = list(speakers_dict.values())

    if args.list_speakers:
        print(f"Total speakers: {n_speakers}")
        print("Speaker name -> ID (first 30):")
        for i, (name, sid) in enumerate(list(speakers_dict.items())[:30]):
            print(f"  {name} -> {sid}")
        print("  ...")
        return

    if args.speaker_id < 0 or args.speaker_id >= n_speakers:
        print(f"Error: speaker_id must be in [0, {n_speakers - 1}]. Got {args.speaker_id}.")
        sys.exit(1)

    # Model and vocoder
    class Args:
        restore_step = args.restore_step
    model = get_model(Args(), configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    language = preprocess_config["preprocessing"]["text"].get("language", "en")
    result = infer(
        model,
        vocoder,
        configs,
        args.text,
        args.speaker_id,
        pitch_control=args.pitch_control,
        energy_control=args.energy_control,
        duration_control=args.duration_control,
        language=language,
    )

    # Output basename
    if args.output_name:
        basename = args.output_name
    else:
        basename = re.sub(r"[^\w\s-]", "", args.text)[:50].strip() or "output"
        basename = re.sub(r"[-\s]+", "_", basename)

    os.makedirs(args.output_dir, exist_ok=True)
    wav_path = os.path.join(args.output_dir, f"{basename}.wav")
    wavfile.write(wav_path, result["sampling_rate"], result["wav"])
    print(f"Saved: {wav_path}")

    if args.save_fig and result.get("mel") is not None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from utils.tools import plot_mel, expand
        preprocess_path = result["stats_path"]
        with open(os.path.join(preprocess_path, "stats.json"), "r") as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]
        pitch = result["pitch"]
        energy = result["energy"]
        duration = result["duration"]
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = expand(pitch, duration)
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = expand(energy, duration)
        mel_len = result["mel"].shape[1]
        pitch = pitch[:mel_len] if len(pitch) > mel_len else np.pad(pitch, (0, mel_len - len(pitch)))
        energy = energy[:mel_len] if len(energy) > mel_len else np.pad(energy, (0, mel_len - len(energy)))
        fig = plot_mel(
            [(result["mel"], pitch, energy)],
            stats,
            ["Synthesized Spectrogram"],
        )
        fig_path = os.path.join(args.output_dir, f"{basename}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved figure: {fig_path}")

    # Report timing and speaker
    print("\n--- Generation time ---")
    print(f"  Speaker ID:        {args.speaker_id} (of {n_speakers} speakers)")
    print(f"  Model forward:     {result['time_model_sec']:.4f} s")
    print(f"  Vocoder:           {result['time_vocoder_sec']:.4f} s")
    print(f"  Total:             {result['time_total_sec']:.4f} s")
    print(f"  Audio duration:    {result['audio_duration_sec']:.4f} s")
    print(f"  RTF:               {result['rtf']:.4f} (real-time factor)")


def launch_gui():
    """Launch Gradio UI. Install with: pip install gradio"""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required for the GUI. Install with: pip install gradio")
        sys.exit(1)

    # Load configs and models once (we're already in FASTSPEECH2_DIR)
    preprocess_config = yaml.load(
        open("config/LibriTTS/preprocess.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("config/LibriTTS/model.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("config/LibriTTS/train.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)
    speakers_dict = load_speakers(preprocess_config)
    n_speakers = len(speakers_dict)

    class Args:
        restore_step = 800000
    print("Loading model and vocoder...")
    model = get_model(Args(), configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    language = preprocess_config["preprocessing"]["text"].get("language", "en")

    def synthesize_ui(text, speaker_id, pitch, energy, duration_speed):
        if not text or not text.strip():
            return None, "Enter some text."
        speaker_id = int(speaker_id)
        if speaker_id < 0 or speaker_id >= n_speakers:
            return None, f"Speaker ID must be 0-{n_speakers - 1}."
        try:
            result = infer(
                model, vocoder, configs, text.strip(), speaker_id,
                pitch_control=float(pitch), energy_control=float(energy),
                duration_control=float(duration_speed), language=language,
            )
            sr = result["sampling_rate"]
            msg = (
                f"**Generation time**  \n"
                f"Model: {result['time_model_sec']:.3f} s  \n"
                f"Vocoder: {result['time_vocoder_sec']:.3f} s  \n"
                f"Total: {result['time_total_sec']:.3f} s  \n"
                f"Audio length: {result['audio_duration_sec']:.3f} s  \n"
                f"RTF: {result['rtf']:.3f}"
            )
            return (sr, result["wav"]), msg
        except Exception as e:
            return None, f"Error: {e}"

    with gr.Blocks(title="FastSpeech2 TTS", theme=gr.themes.Soft()) as app:
        gr.Markdown("# FastSpeech2 Multi-Speaker TTS")
        with gr.Row():
            text_in = gr.Textbox(
                label="Text to synthesize",
                placeholder="Enter English text here...",
                value="Hello world",
                lines=3,
            )
        with gr.Row():
            speaker_id_in = gr.Number(
                value=0,
                minimum=0,
                maximum=n_speakers - 1,
                step=1,
                label=f"Speaker ID (0–{n_speakers - 1})",
                precision=0,
            )
        with gr.Row():
            pitch_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Pitch (1.0 = normal)")
            energy_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Energy (1.0 = normal)")
            duration_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed (1.0 = normal)")
        btn = gr.Button("Generate speech", variant="primary")
        audio_out = gr.Audio(label="Output", type="numpy")
        time_md = gr.Markdown(label="Generation time")

        btn.click(
            fn=synthesize_ui,
            inputs=[text_in, speaker_id_in, pitch_slider, energy_slider, duration_slider],
            outputs=[audio_out, time_md],
        )

    print("Opening Gradio UI in browser...")
    app.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
