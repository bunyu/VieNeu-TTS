"""Turn any voice of the web app into a voice for the "Doc truyen" phone app.

The phone reads with VieNeu-TTS v3 Nano, which keeps a voice as a speaker vector (192 numbers) and
style tokens (50 x 256); a Turbo voice, cloned or fine-tuned, is kept another way and cannot be
moved across as it is. So the chosen voice reads a passage here, and Nano enrols that speech, as the
phone would enrol a recording. The result is a ``.dtvoice`` file (a zip):

    voice.json   {"format": "doc-truyen-voice", "version": 1, "name", "description", "release",
                  "speaker": [192 floats], "style": [[256 floats] x 50]}
    sample.wav   the converted voice read by Nano: what the phone will sound like

Nano must be the release the phone pins (``android/app/src/main/assets/vieneu-release.json``), or
the numbers would mean something else to the phone's model. Its files are fetched once into
``.hf-home/phone-nano`` and checked against that manifest; ``VIENEU_NANO_DIR`` may name a folder
that already holds them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "android" / "app" / "src" / "main" / "assets" / "vieneu-release.json"
SUFFIX = ".dtvoice"
# What Nano needs to speak and to enrol; the denoiser is not needed (the speech it enrols is clean).
NEEDED = ["text_encoder.onnx", "duration_predictor.onnx", "vector_estimator.onnx", "codec_decoder.onnx", "config.json", "constants.npz",
          "speaker_encoder.onnx", "codec_encoder.onnx", "reference_encoder.onnx"]
# About ten seconds of plain narration: the passage the twenty extra phone voices were enrolled from.
PASSAGE = ("Buổi sáng hôm ấy, trời trong xanh và gió nhẹ thổi qua những hàng cây ven hồ. "
           "Bà cụ ngồi bên hiên nhà, chậm rãi kể cho các cháu nghe câu chuyện cũ về ngôi làng ngày xưa.")
CHECK = "Xin chào, tôi sẽ đọc sách cho bạn nghe bằng giọng này. Chúc bạn có những phút giây thật thư giãn."


def release() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nano_dir(report: Callable[[str], None] = print) -> Path:
    """A folder holding the phone's Nano release, downloading what is missing (about 380 MB the first time)."""
    files = {f["name"]: f for f in release()["files"]}
    given = os.getenv("VIENEU_NANO_DIR")
    if given and all((Path(given) / n).is_file() and (Path(given) / n).stat().st_size == files[n]["size"] for n in NEEDED):
        return Path(given)
    home = Path(os.getenv("HF_HOME") or ROOT / ".hf-home") / "phone-nano"
    home.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        entry, target = files[name], home / name
        if target.is_file() and target.stat().st_size == entry["size"]:
            continue
        report(f"Đang tải {name} ({entry['size'] / 1e6:.0f} MB) của model điện thoại…")
        pending = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(entry["url"], pending)
        if _sha256(pending) != entry["sha256"]:
            pending.unlink(missing_ok=True)
            raise RuntimeError(f"{name} tải về không khớp mã kiểm tra; thử lại sau.")
        pending.replace(target)
    return home


_nano = None


def load_nano(report: Callable[[str], None] = print):
    """The Nano engine of the phone's release, loaded once (on the CPU, beside whatever the web app runs)."""
    global _nano
    if _nano is None:
        from vieneu.v3nano import OnnxV3NanoEngine
        _nano = OnnxV3NanoEngine(local_dir=str(nano_dir(report)), threads=4)
    return _nano


def safe_file_name(name: str) -> str:
    return " ".join(re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name).split()).strip(" .")[:60] or "giong"


def export(tts, voice_id: Optional[str], name: str, out_dir, *, description: str = "", nano=None,
           report: Callable[[str], None] = print) -> dict:
    """Write ``<name>.dtvoice`` for the voice ``voice_id`` of the loaded model ``tts`` (None: its default voice).

    Returns ``file`` (the .dtvoice), ``sample`` (the converted voice reading a line, as the phone will)
    and ``source`` (the original voice reading the passage it was enrolled from).
    """
    name = " ".join((name or "").split())[:60]
    if not name:
        raise ValueError("Hãy đặt tên cho giọng trên điện thoại.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="vieneu-phone-"))
    report("Giọng đang đọc đoạn mẫu…")
    kwargs = {"voice": tts.get_preset_voice(voice_id)} if voice_id else {}
    spoken = np.asarray(tts.infer(PASSAGE, **kwargs), dtype=np.float32).reshape(-1)
    if len(spoken) < 2 * tts.sample_rate:
        raise ValueError("Giọng này không đọc ra được đoạn mẫu.")
    source = work / "source.wav"
    sf.write(str(source), spoken, tts.sample_rate)
    report("Model điện thoại đang học giọng…")
    nano = nano or load_nano(report)
    # The speech is clean, so the denoiser would only colour it.
    speaker, style = nano.prepare_reference(str(source), denoise=False)
    speaker = np.asarray(speaker, np.float32).reshape(-1)
    style = np.asarray(style, np.float32)
    if speaker.shape != (192,) or style.shape != (50, 256):
        raise RuntimeError(f"Kết quả không đúng dạng ({speaker.shape}, {style.shape}).")
    report("Đọc thử bằng giọng trên điện thoại…")
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions
    check = np.asarray(nano.infer(phonemize_text_with_emotions(CHECK), speaker, style, steps=16, cfg=3.0, seed=0), dtype=np.float32).reshape(-1)
    sample = work / "sample.wav"
    sf.write(str(sample), check, 24000, subtype="PCM_16")
    data = {"format": "doc-truyen-voice", "version": 1, "name": name, "description": description.strip(),
            "release": release()["id"], "speaker": [round(float(x), 6) for x in speaker],
            "style": [[round(float(x), 5) for x in row] for row in style]}
    target = out / (safe_file_name(name) + SUFFIX)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("voice.json", json.dumps(data, ensure_ascii=False))
        z.write(sample, "sample.wav")
    return {"file": str(target), "sample": str(sample), "source": str(source)}
