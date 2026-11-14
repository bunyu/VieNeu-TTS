"""Measure VieNeu-TTS speed on this PC: v3 Turbo on the GPU (PyTorch) and on the CPU (ONNX).

Usage: python apps/bench_windows.py [--cpu] [--out DIR]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TEXT = ("Buổi sáng hôm ấy, trời trong xanh và gió nhẹ thổi qua những hàng cây ven hồ. "
        "Bà cụ ngồi bên hiên nhà, kể cho các cháu nghe câu chuyện cũ về ngôi làng ngày xưa.")


def run(label, tts, out_dir, voice=None):
    kwargs = {"voice": tts.get_preset_voice(voice)} if voice else {}
    tts.infer("Xin chào.", **kwargs)  # warm-up (CUDA graphs, caches)
    start = time.perf_counter()
    audio = np.asarray(tts.infer(TEXT, **kwargs), dtype=np.float32)
    took = time.perf_counter() - start
    seconds = len(audio) / tts.sample_rate
    print(f"{label}: {seconds:.2f} s of audio in {took:.2f} s (real-time factor {took / seconds:.3f})")
    sf.write(out_dir / f"{label}.wav", audio, tts.sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="also time the ONNX/CPU backend")
    parser.add_argument("--out", default="bench-output")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from vieneu import Vieneu
    from vieneu.gpu_support import turbo_gpu_kwargs
    kwargs = turbo_gpu_kwargs()
    print("GPU settings:", kwargs or "none (no CUDA GPU)")
    if kwargs:
        run("turbo-gpu", Vieneu("v3turbo", **kwargs), out)
    if args.cpu or not kwargs:
        run("turbo-cpu", Vieneu("v3turbo", backend="onnx"), out)


if __name__ == "__main__":
    main()
