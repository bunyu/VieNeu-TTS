"""Measure how much a v3 Nano clone sounds like its reference recording.

For each reference clip, enrol it with several recipes, speak a fixed Vietnamese passage with the
cloned voice, and report the cosine similarity between the speaker vector of the generated speech
and that of the original (clean) recording: what a listener compares. Optionally adds room noise to
the reference first, like a phone recording.

    python tests/clone_similarity.py --model DIR [--noise-db 20] [--refs a.wav b.wav ...]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import soundfile as sf

AUTO_DENOISE_BELOW_DB = 35.0

TEXT = ("Buổi sáng hôm ấy, trời trong xanh và gió nhẹ thổi qua những hàng cây ven hồ. "
        "Bà cụ ngồi bên hiên nhà, kể cho các cháu nghe câu chuyện cũ.")


def most_voiced_window(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """The ``seconds``-long stretch with the most speech energy (20 ms frames above -35 dB of the peak)."""
    n = int(seconds * sr)
    if len(wav) <= n:
        return wav
    hop = sr // 50
    frames = len(wav) // hop
    energy = np.array([np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(frames)])
    voiced = (20 * np.log10(energy / energy.max()) > -35).astype(np.float64)
    width = n // hop
    scores = np.convolve(voiced, np.ones(width), mode="valid")
    start = int(np.argmax(scores)) * hop
    return wav[start:start + n]


def snr_db(wav: np.ndarray, sr: int) -> float:
    """Rough signal-to-noise ratio: loud (90th percentile) vs quiet (10th percentile) 20 ms frames."""
    hop = sr // 50
    frames = np.array([np.sqrt(np.mean(wav[i:i + hop] ** 2) + 1e-12) for i in range(0, len(wav) - hop, hop)])
    return float(20 * np.log10(np.percentile(frames, 90) / np.percentile(frames, 10)))


def enrol(engine, wav, sr, *, denoise, window):
    from vieneu._v3_turbo_engine.speaker.audio_utils import high_quality_resample
    if denoise == "auto":
        denoise = snr_db(wav, sr) < AUTO_DENOISE_BELOW_DB
    if denoise:
        den = engine._get_denoiser()
        wav = np.asarray(den.denoise(wav, sr), dtype=np.float32)
        sr = 44100
    spk = engine.speaker_encoder.embed(wav, sr)
    m24 = wav if sr == 24000 else np.asarray(high_quality_resample(wav, sr, 24000), dtype=np.float32)
    m24 = most_voiced_window(m24, 24000, 5.0) if window else m24[:24000 * 5]
    mu = engine.s_codec_enc.run(None, {engine.s_codec_enc.get_inputs()[0].name: m24[None, None]})[0]
    z = (mu - engine.lat_mean) / engine.lat_std * engine.lat_scale
    z = engine._group_latent(z)[:, :, : min(int(5.0 * engine.fps), engine.ref_max_frames)]
    style = engine.s_ref.run(None, {"ref": z.astype(np.float32), "ref_mask": np.ones((1, z.shape[2]), bool)})[0][0]
    return np.asarray(spk, np.float32), np.asarray(style, np.float32)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--refs", nargs="*")
    p.add_argument("--noise-db", type=float, default=None, help="add white+hum noise at this SNR")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default=None, help="write the generated audio here")
    args = p.parse_args()
    from sea_g2p import G2P  # noqa: F401  (import check)
    from vieneu.v3nano import OnnxV3NanoEngine
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    engine = OnnxV3NanoEngine(local_dir=args.model)
    engine._ensure_clone_sessions()
    phones = phonemize_text_with_emotions(TEXT)
    refs = args.refs or sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "examples/audio_ref/*.wav")))
    recipes = {
        "as phone app (no denoise, first 5 s)": dict(denoise=False, window=False, steps=16, cfg=3.0),
        "+ denoise": dict(denoise=True, window=False, steps=16, cfg=3.0),
        "+ denoise + best 5 s": dict(denoise=True, window=True, steps=16, cfg=3.0),
        "+ denoise + best 5 s, cfg 4": dict(denoise=True, window=True, steps=16, cfg=4.0),
        "+ denoise + best 5 s, 8 steps sway -1 (fast)": dict(denoise=True, window=True, steps=8, cfg=3.0, sway=-1.0),
        "no denoise, 8 steps (fast, today)": dict(denoise=False, window=False, steps=8, cfg=3.0, sway=-1.0),
        "denoise only when noisy": dict(denoise="auto", window=True, steps=16, cfg=3.0),
    }
    rng = np.random.default_rng(0)
    results = {k: [] for k in recipes}
    for ref in refs:
        clean, sr = sf.read(ref, dtype="float32", always_2d=True)
        clean = clean.mean(axis=1)
        target = engine.speaker_encoder.embed(clean, sr)
        heard = clean
        if args.noise_db is not None:
            t = np.arange(len(clean)) / sr
            noise = rng.normal(size=len(clean)).astype(np.float32) + 2 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
            gain = np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2)) / (10 ** (args.noise_db / 20))
            heard = clean + gain * noise
        print(f"  {Path(ref).name}: estimated SNR {snr_db(heard, sr):.1f} dB")
        for name, r in recipes.items():
            spk, style = enrol(engine, heard, sr, denoise=r["denoise"], window=r["window"])
            sims = []
            for seed in range(args.seeds):
                wav = engine.infer(phones, spk, style, steps=r["steps"], cfg=r["cfg"], sway=r.get("sway", 0.0), seed=seed)
                sims.append(cosine(engine.speaker_encoder.embed(wav, 24000), target))
                if args.out and seed == 0:
                    Path(args.out).mkdir(parents=True, exist_ok=True)
                    sf.write(Path(args.out) / f"{Path(ref).stem} - {name.replace('/', '-')}.wav", wav, 24000)
            results[name].append(float(np.mean(sims)))
        print(Path(ref).name, " ".join(f"{np.mean(v[-1:]):.3f}" for v in results.values()))
    print(f"\nmean similarity of generated speech to the reference ({len(refs)} clips x {args.seeds} seeds"
          + (f", reference noise SNR {args.noise_db} dB" if args.noise_db is not None else "") + "):")
    for name, v in results.items():
        print(f"  {np.mean(v):.3f}  {name}")


if __name__ == "__main__":
    main()
