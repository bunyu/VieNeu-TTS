"""Make the reader's extra voices: the preset voices of VieNeu-TTS v3 Turbo, enrolled for v3 Nano.

v3 Nano ships 11 voices and v3 Turbo 25. A Turbo voice cannot be loaded into Nano (it is stored as
Turbo codec frames), so each Turbo voice first speaks a reference passage, and Nano then enrols
that speech exactly as it enrols a recording (`encode_reference`). The result is a Nano preset file
that the app packs as `assets/voices_extra.json`; names Nano already has are left to Nano.

Two steps, run one after the other because this PC cannot hold both models at once:

    .venv\\Scripts\\python android\\app\\scripts\\make_extra_voices.py speak --work D:\\work\\audio\\extra-voices
    .venv\\Scripts\\python android\\app\\scripts\\make_extra_voices.py enrol --work D:\\work\\audio\\extra-voices --nano D:\\work\\audio\\vieneu-nano
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "src" / "vieneu" / "assets"
OUT = REPO / "android" / "app" / "src" / "main" / "assets" / "voices_extra.json"
# About ten seconds of plain narration: enough for the speaker vector, and Nano takes its style from the most voiced 5 s.
PASSAGE = ("Buổi sáng hôm ấy, trời trong xanh và gió nhẹ thổi qua những hàng cây ven hồ. "
           "Bà cụ ngồi bên hiên nhà, chậm rãi kể cho các cháu nghe câu chuyện cũ về ngôi làng ngày xưa.")


def presets(name):
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))["presets"]


def wanted():
    """Turbo voices Nano does not already have, in Turbo's order."""
    nano = presets("voices_v3_nano.json")
    return [(i, name, v) for i, (name, v) in enumerate(presets("voices_v3_turbo.json").items()) if name not in nano]


def speak(work: Path):
    from vieneu import Vieneu
    from vieneu.gpu_support import turbo_gpu_kwargs
    tts = Vieneu("v3turbo", **(turbo_gpu_kwargs() or {"backend": "onnx"}))
    work.mkdir(parents=True, exist_ok=True)
    for i, name, _ in wanted():
        target = work / f"{i:02d}.wav"
        if target.exists():
            continue
        audio = np.asarray(tts.infer(PASSAGE, voice=tts.get_preset_voice(name)), dtype=np.float32)
        sf.write(target, audio, tts.sample_rate)
        print(f"{i:02d} {name}: {len(audio) / tts.sample_rate:.1f} s", flush=True)


def enrol(work: Path, nano_dir: str):
    from vieneu import Vieneu
    tts = Vieneu("v3nano", onnx_dir=nano_dir)
    result = {}
    for i, name, voice in wanted():
        # The reference is clean synthetic speech, so the denoiser would only colour it.
        speaker, style = tts.encode_reference(str(work / f"{i:02d}.wav"), denoise=False)
        result[name] = {
            "description": voice.get("description", ""), "gender": voice.get("gender", ""),
            "speaker_emb": [round(float(x), 6) for x in np.asarray(speaker).reshape(-1)],
            "style": [[round(float(x), 5) for x in row] for row in np.asarray(style)],
        }
        # Proof by ear: the enrolled voice reading a line.
        check = np.asarray(tts.infer("Xin chào, tôi sẽ đọc sách cho bạn nghe bằng giọng này.", voice={"speaker_emb": speaker, "style": style}), dtype=np.float32)
        sf.write(work / f"{i:02d}-nano.wav", check, tts.sample_rate)
        print(f"{i:02d} {name}: enrolled", flush=True)
    data = {"meta": {"note": "VieNeu-TTS v3 Turbo preset voices enrolled for v3 Nano by make_extra_voices.py"}, "default_voice": None, "presets": result}
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{len(result)} voices -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["speak", "enrol"])
    parser.add_argument("--work", required=True)
    parser.add_argument("--nano", help="folder holding the v3 Nano release, with its cloning graphs")
    args = parser.parse_args()
    if args.step == "speak":
        speak(Path(args.work))
    else:
        enrol(Path(args.work), args.nano)
