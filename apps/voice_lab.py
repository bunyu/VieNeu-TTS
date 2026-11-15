"""Find the best clone of a voice from one longer recording.

A clone is made from at most eight seconds of audio, and those seconds decide both the timbre and
the way of reading. Given a recording of half a minute or more, this picks several eight-second
stretches full of speech, clones each, has every clone read the same passage, and scores how much
the result sounds like the speaker of the whole recording (cosine similarity of speaker vectors).
The score orders the candidates; the ear decides.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

CLIP_SECONDS = 8.0
PASSAGE = ("Buổi sáng hôm ấy, trời trong xanh và gió nhẹ thổi qua những hàng cây ven hồ. "
           "Bà cụ ngồi bên hiên nhà, chậm rãi kể cho các cháu nghe câu chuyện cũ về ngôi làng ngày xưa.")


def load_mono(path) -> tuple:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def voiced_frames(wav: np.ndarray, sr: int) -> tuple:
    """Which 20 ms frames hold speech (within 35 dB of the loudest), and the frame length in samples."""
    hop = max(1, sr // 50)
    count = len(wav) // hop
    if count == 0:
        return np.zeros(0, bool), hop
    energy = np.sqrt(np.mean(wav[:count * hop].reshape(count, hop) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(energy / max(float(energy.max()), 1e-9)) > -35, hop


def candidate_windows(wav: np.ndarray, sr: int, seconds: float = CLIP_SECONDS, count: int = 4) -> List[tuple]:
    """Up to ``count`` stretches of ``seconds`` that are nearly all speech and overlap by half at most,
    each moved to start where a phrase starts; returned in the order they occur, as sample ranges."""
    size = int(seconds * sr)
    if len(wav) <= size:
        return [(0, len(wav))]
    voiced, hop = voiced_frames(wav, sr)
    width = size // hop
    if len(voiced) < width:
        return [(0, len(wav))]
    score = np.convolve(voiced.astype(np.float64), np.ones(width), mode="valid") / width
    free = np.ones(len(score), bool)
    best = float(score.max())
    chosen: List[int] = []
    while len(chosen) < count and free.any():
        at = int(np.argmax(np.where(free, score, -1.0)))
        if score[at] < 0.85 * best:
            break  # what is left has too much silence in it
        # Start on the first word after a pause, so the clone does not begin inside a word.
        start = at
        for f in range(at, min(at + 50, len(score) - 1)):
            if voiced[f] and f >= 3 and not voiced[f - 3:f].any():
                start = max(f - 2, 0)
                break
        chosen.append(start)
        # Candidates may share up to half their length, so a recording of half a minute still yields several.
        free[max(0, at - width // 2):at + width // 2] = False
    return [(s * hop, min(s * hop + size, len(wav))) for s in sorted(chosen)]


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compare(tts, audio_path, out_dir, text: str = PASSAGE, count: int = 4, both_denoise: bool = False,
            report: Optional[Callable[[str], None]] = None) -> List[dict]:
    """Clone every candidate stretch of ``audio_path`` and return them, most like the speaker first.

    Each result holds ``clip`` (the stretch, a wav file to save the voice from), ``sample`` (the clone
    reading ``text``), ``start`` / ``end`` in seconds, ``denoise`` and ``similarity``.
    """
    engine = getattr(tts, "engine", None)
    if engine is None or not hasattr(engine, "extract_speaker_emb") or not hasattr(tts, "encode_reference"):
        raise ValueError("So sánh đoạn mẫu chỉ dùng được với VieNeu v3 Turbo.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav, sr = load_mono(audio_path)
    if len(wav) < 3 * sr:
        raise ValueError("Bản thu quá ngắn. Cần ít nhất vài giây giọng nói, tốt nhất là 30 giây trở lên.")
    # What "the speaker" sounds like: up to a minute of the recording, not any single stretch of it.
    whole = out / "whole.wav"
    sf.write(str(whole), wav[:60 * sr], sr)
    target = engine.extract_speaker_emb(str(whole))
    results: List[dict] = []
    windows = candidate_windows(wav, sr, count=count)
    modes = (True, False) if both_denoise else (True,)
    for n, (a, b) in enumerate(windows, 1):
        clip = out / f"clip-{n}.wav"
        sf.write(str(clip), wav[a:b], sr)
        for denoise in modes:
            if report:
                report(f"Đoạn {n}/{len(windows)} ({a / sr:.0f}–{b / sr:.0f} giây), {'có' if denoise else 'không'} khử nhiễu…")
            speaker, codes = tts.encode_reference(str(clip), denoise=denoise)
            spoken = np.asarray(tts.infer(text, voice={"speaker_emb": speaker, "codes": codes}), dtype=np.float32).reshape(-1)
            sample = out / f"sample-{n}-{'denoise' if denoise else 'raw'}.wav"
            sf.write(str(sample), spoken, tts.sample_rate)
            results.append({"clip": str(clip), "sample": str(sample), "start": a / sr, "end": b / sr, "denoise": denoise,
                            "similarity": cosine(engine.extract_speaker_emb(str(sample)), target)})
    return sorted(results, key=lambda r: -r["similarity"])


def label(rank: int, r: dict) -> str:
    return (f"#{rank} · giây {r['start']:.0f}–{r['end']:.0f} · {'có' if r['denoise'] else 'không'} khử nhiễu · "
            f"độ giống {r['similarity']:.2f}")
