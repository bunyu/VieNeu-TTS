"""Record a fine-tuning dataset for one voice, sentence by sentence, and train on it.

Fine-tuning (finetune/README.md) needs 10 to 30 minutes of one speaker as clips of 3 to 15 seconds,
each with its exact text. Reading known sentences one at a time gives exactly that with no
transcription step: the web app shows a sentence, the speaker records it, and this module files the
take under ``finetune/dataset/<voice>/raw_audio`` and keeps ``metadata.csv`` in step.

Training is a separate run, because a 6 GB card cannot hold the app's model and a training run
together:

    python -m apps.finetune_studio train "<voice>"        (or finetune_windows.bat "<voice>")

It prepares the dataset, trains the LoRA with settings chosen for this card, merges it, and enrols
the voice into the merged model, which the web app then lists as a model of its own.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "finetune" / "dataset"
OUTPUT = ROOT / "finetune" / "output"
MIN_SECONDS, MAX_SECONDS = 3.0, 15.0
TARGET_MINUTES = 15.0

# Read when no book is given: everyday narration, with questions, numbers and dialogue for variety.
BUILT_IN = """Buổi sáng hôm ấy, trời trong xanh và gió nhẹ thổi qua những hàng cây ven hồ.
Bà cụ ngồi bên hiên nhà, chậm rãi kể cho các cháu nghe câu chuyện cũ về ngôi làng ngày xưa.
Anh dừng lại trước cổng, ngập ngừng một lúc rồi mới đưa tay gõ cửa.
Cô bé hỏi mẹ: "Bao giờ thì bố về hả mẹ?"
Không ai trả lời, chỉ có tiếng mưa rơi đều đều trên mái ngói.
Năm đó tôi vừa tròn mười tám tuổi, lần đầu tiên xa nhà lên thành phố học.
Chuyến tàu khởi hành lúc sáu giờ ba mươi phút sáng và đến nơi vào khoảng giữa trưa.
Ông lão mỉm cười, đôi mắt nheo lại sau cặp kính dày.
"Cậu cứ yên tâm, mọi chuyện rồi sẽ ổn cả thôi," bà nói khẽ.
Con đường làng quanh co chạy giữa hai hàng tre xanh mướt.
Tôi chưa bao giờ nghĩ rằng mình sẽ quay lại nơi này sau ngần ấy năm.
Trong bếp, nồi canh chua đang sôi, mùi thơm lan ra tận ngoài sân.
Thằng bé chạy ùa ra, ôm chầm lấy chân bố, cười khanh khách.
Họ đã đi bộ suốt ba ngày liền mà vẫn chưa thấy bóng dáng một ngôi nhà nào.
Nếu hôm ấy tôi không lên tiếng, có lẽ mọi chuyện đã khác đi rất nhiều.
Chị đặt tách trà xuống bàn, nhìn ra cửa sổ, im lặng hồi lâu.
Giá gạo năm nay tăng gần hai mươi phần trăm so với cùng kỳ năm ngoái.
Đêm xuống, cả khu rừng chìm trong một thứ im lặng nặng nề.
"Sao em không nói với anh sớm hơn?" anh hỏi, giọng run run.
Mùa thu, lá vàng rơi đầy lối đi dẫn vào ngôi trường cũ.
Người lái đò già kể rằng con sông này chưa bao giờ cạn nước.
Cô giáo bước vào lớp, cả phòng lập tức im phăng phắc.
Chúng tôi ngồi quanh đống lửa, nướng khoai và kể chuyện ma cho nhau nghe.
Bức thư chỉ có vài dòng ngắn ngủi, nhưng anh đọc đi đọc lại không biết bao nhiêu lần.
Tiếng gà gáy đầu ngõ báo hiệu một ngày mới lại bắt đầu.
Thật ra, tôi đã biết chuyện đó từ lâu, chỉ là không muốn nói ra.
Ngoài kia, phố xá đã lên đèn, dòng người hối hả trở về nhà sau giờ tan tầm.
Cậu bé nhặt viên sỏi lên, ngắm nghía một hồi rồi bỏ vào túi áo.
Bà ngoại tôi năm nay đã ngoài tám mươi nhưng vẫn còn minh mẫn lắm.
"Đừng lo, tôi sẽ ở lại đây cho đến khi anh ấy quay về."
Cơn bão đi qua, để lại những mái nhà tốc ngói và hàng cây đổ rạp.
Anh ta nhìn đồng hồ, thở dài, rồi lặng lẽ đứng dậy ra về.
Mỗi buổi chiều, lũ trẻ lại kéo nhau ra bãi cỏ ven đê thả diều.
Trên bàn vẫn còn nguyên bữa cơm chưa ai động đũa.
Có những điều, phải đến khi mất đi rồi người ta mới hiểu hết giá trị của nó.
Chiếc xe buýt cuối cùng rời bến lúc mười giờ kém mười lăm.
Cô khẽ gật đầu, không nói gì thêm, nhưng ánh mắt đã sáng lên.
Mẹ dặn tôi phải mặc ấm vì ngoài trời đang trở gió.
Ông chủ quán mang ra hai bát phở nóng hổi, khói bay nghi ngút.
Và thế là, sau bao nhiêu năm xa cách, hai anh em lại được ngồi bên nhau."""


def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", (name or "").strip())
    name = " ".join(name.split()).strip(" .")[:40]
    if not name:
        raise ValueError("Hãy đặt tên cho giọng.")
    return name


def dataset_dir(name: str) -> Path:
    return DATASETS / safe_name(name)


def split_sentences(text: str, shortest: int = 35, longest: int = 170) -> List[str]:
    """Sentences fit for one take: long enough to last three seconds, short enough to read in one breath."""
    out: List[str] = []
    for paragraph in (p.strip() for p in text.split("\n")):
        if not paragraph:
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?…])[\"')\]»”’]*\s+(?=[\"'“‘(«\-–—]?[A-ZÀ-ỸĐ0-9])", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            joined = f"{current} {sentence}".strip()
            if len(joined) <= longest:
                current = joined
                if len(current) >= shortest:
                    out.append(current)
                    current = ""
            else:
                if len(current) >= shortest:
                    out.append(current)
                current = sentence if len(sentence) <= longest else ""
                if len(current) >= shortest:
                    out.append(current)
                    current = ""
    return out


def load_script(source: Optional[str] = None, limit: int = 400) -> List[str]:
    """The sentences to read: from a book or text file when given (its own style of prose), else the built-in ones."""
    if source and str(source).strip():
        from apps.book_batch import collect_parts
        text = "\n".join(part.text for part in collect_parts(Path(str(source).strip().strip('"'))))
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("Không tìm được câu nào phù hợp trong file đó.")
        return sentences[:limit]
    return [line.strip() for line in BUILT_IN.splitlines() if line.strip()]


def _takes(folder: Path) -> dict:
    try:
        return json.loads((folder / "takes.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write(folder: Path, takes: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "takes.json").write_text(json.dumps(takes, ensure_ascii=False, indent=1), encoding="utf-8")
    # What prepare_dataset.py reads: file_name|text, one clip per line.
    lines = [f"{take['file']}|{text}" for text, take in sorted(takes.items(), key=lambda kv: kv[1]["file"])]
    (folder / "metadata.csv").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_take(name: str, text: str, audio_path: str) -> float:
    """File one recorded sentence (a second take of the same sentence replaces the first); returns its seconds."""
    text = " ".join((text or "").split())
    if "|" in text or not text:
        raise ValueError("Câu này không dùng được (trống hoặc có ký tự |).")
    if not audio_path:
        raise ValueError("Chưa có bản thu. Bấm thu âm, đọc câu, rồi dừng.")
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    # Trim the silence before and after: reaching for the button is not part of the sentence.
    loud = np.flatnonzero(np.abs(wav) > 0.02 * max(float(np.abs(wav).max()), 1e-6))
    if len(loud):
        wav = wav[max(0, loud[0] - sr // 5):loud[-1] + sr // 4]
    seconds = len(wav) / sr
    if seconds < MIN_SECONDS:
        raise ValueError(f"Bản thu chỉ dài {seconds:.1f} giây; cần ít nhất {MIN_SECONDS:.0f} giây. Hãy đọc lại chậm rãi hơn.")
    if seconds > MAX_SECONDS:
        raise ValueError(f"Bản thu dài {seconds:.1f} giây; tối đa {MAX_SECONDS:.0f} giây. Hãy bỏ qua câu này.")
    folder = dataset_dir(name)
    takes = _takes(folder)
    file = takes.get(text, {}).get("file") or f"{max([int(Path(t['file']).stem) for t in takes.values()] + [0]) + 1:04d}.wav"
    (folder / "raw_audio").mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / "raw_audio" / file), wav, sr, subtype="PCM_16")
    takes[text] = {"file": file, "seconds": round(seconds, 2)}
    _write(folder, takes)
    return seconds


def progress(name: str) -> tuple:
    """(sentences recorded, minutes recorded, the texts recorded) for the voice ``name``."""
    takes = _takes(dataset_dir(name))
    return len(takes), sum(t["seconds"] for t in takes.values()) / 60, set(takes)


def merged_models() -> List[tuple]:
    """Fine-tuned models ready to load: (run name, folder)."""
    if not OUTPUT.is_dir():
        return []
    return sorted((p.parent.name, str(p)) for p in OUTPUT.glob("*/merged") if (p / "config.json").is_file() or any(p.glob("**/config.json")))


# ── Training ─────────────────────────────────────────────────────────────────

def training_command(name: str, epochs: float = 3.0) -> List[str]:
    """train_lora.py arguments for this PC's card."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Fine-tune cần GPU NVIDIA (CUDA).")
    capability = torch.cuda.get_device_capability(0)
    gigabytes = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    # No --merge: merging inside the training process runs out of memory on an 8 GB PC; train() merges in a process of its own.
    args = ["--data", str(dataset_dir(name) / "train.parquet"), "--run", safe_name(name), "--epochs", str(epochs), "--num-workers", "0"]
    if capability[0] < 8:
        args.append("--no-bf16")  # no bfloat16 before RTX 30: stay in float32, as the app does on these cards
    if gigabytes < 7 or capability[0] < 8:
        # Float32 on a 6 GB card: small batches, the same 16 clips per step, and less memory per clip.
        args += ["--batch-size", "2", "--grad-accum", "8", "--grad-checkpoint"]
    return args


def train(name: str, epochs: float = 3.0) -> None:
    folder = dataset_dir(name)
    count, minutes, _ = progress(name)
    if count < 20:
        raise SystemExit(f"Mới có {count} câu ({minutes:.1f} phút) trong {folder}. Cần ít nhất 20 câu; nên có 100–300 câu (10–30 phút).")
    print(f"== {safe_name(name)}: {count} câu, {minutes:.1f} phút ==", flush=True)

    def run(script: str, *args: str) -> None:
        print(f"\n== {script} {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "finetune" / script), *args], check=True, cwd=str(ROOT))

    run("prepare_dataset.py", "--dataset-dir", str(folder))
    run("train_lora.py", *training_command(name, epochs))
    merged = OUTPUT / safe_name(name) / "merged"
    run("merge_lora.py", "--adapter", str(OUTPUT / safe_name(name) / "adapter"), "--out", str(merged))
    # The voice of the merged model is a speaker vector; take it from the longest clip that fits in eight seconds.
    takes = _takes(folder)
    best = max((t for t in takes.values() if t["seconds"] <= 8.0), key=lambda t: t["seconds"], default=None)         or min(takes.values(), key=lambda t: t["seconds"])
    run("make_voice.py", "--audio", str(folder / "raw_audio" / best["file"]), "--name", safe_name(name), "--out", str(merged), "--default")
    print(f"\n✅ Xong. Mở lại app (run_windows.bat), chọn model \"{model_label(safe_name(name))}\" rồi bấm nạp model.", flush=True)


def model_label(run: str) -> str:
    return f"VieNeu-TTS-v3-Turbo 🎓 {run}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train"])
    parser.add_argument("name", help="the voice name used when recording")
    parser.add_argument("--epochs", type=float, default=3.0)
    arguments = parser.parse_args()
    train(arguments.name, arguments.epochs)
