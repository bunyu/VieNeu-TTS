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


PHONE_SENTENCES = ROOT / "android" / "app" / "src" / "main" / "assets" / "finetune_sentences.txt"


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


def load_audio(path) -> tuple:
    """Any recording as mono float32 and its rate: WAV / FLAC / OGG / MP3 directly, phone formats
    (.m4a, .aac, .amr, .3gp, .opus in other containers) through PyAV."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return wav.mean(axis=1), sr
    except Exception:  # noqa: BLE001
        pass
    try:
        import av
    except ImportError as e:
        raise ValueError("Không đọc được file này. Cài thêm gói 'av' (chạy lại setup_windows_gpu.bat) để đọc file ghi âm của điện thoại.") from e
    try:
        with av.open(str(path)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            rate = stream.codec_context.sample_rate or 48000
            resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            chunks = [np.asarray(f.to_ndarray(), np.float32).reshape(-1) for frame in container.decode(stream) for f in resampler.resample(frame)]
            chunks += [np.asarray(f.to_ndarray(), np.float32).reshape(-1) for f in resampler.resample(None)]
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Không đọc được file âm thanh này ({e}).") from e
    if not chunks:
        raise ValueError("File không có âm thanh.")
    return np.concatenate(chunks), rate


def _trimmed(wav: np.ndarray, sr: int) -> np.ndarray:
    """The take without the silence before and after it (reaching for the button is not part of the sentence)."""
    loud = np.flatnonzero(np.abs(wav) > 0.02 * max(float(np.abs(wav).max()), 1e-6))
    return wav[max(0, loud[0] - sr // 5):loud[-1] + sr // 4] if len(loud) else wav


def save_take(name: str, text: str, audio_path: str = None, *, audio: tuple = None) -> float:
    """File one recorded sentence (a second take of the same sentence replaces the first); returns its seconds.
    The take is ``audio_path`` (any format ``load_audio`` reads) or ``audio`` = (samples, rate)."""
    text = " ".join((text or "").split())
    if "|" in text or not text:
        raise ValueError("Câu này không dùng được (trống hoặc có ký tự |).")
    if audio is None and not audio_path:
        raise ValueError("Chưa có bản thu. Bấm thu âm (hoặc tải file lên), đọc câu, rồi dừng.")
    wav, sr = audio if audio is not None else load_audio(audio_path)
    wav = _trimmed(np.asarray(wav, np.float32), sr)
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


def split_reading(wav: np.ndarray, sr: int, gap: float = 0.6) -> List[tuple]:
    """Where the sentences of one long reading are: stretches of speech separated by at least ``gap``
    seconds of quiet, as sample ranges. Pauses at commas are shorter and stay inside a sentence."""
    hop = max(1, sr // 100)                          # 10 ms
    count = len(wav) // hop
    if count == 0:
        return []
    energy = np.sqrt(np.mean(wav[:count * hop].reshape(count, hop) ** 2, axis=1) + 1e-12)
    # Quiet means well below the loud parts of this recording: phone recordings carry a noise floor.
    floor = max(np.percentile(energy, 95) * 10 ** (-30 / 20), np.percentile(energy, 10) * 2)
    loud = energy > floor
    runs, start, quiet = [], None, 0
    for i, on in enumerate(loud):
        if on:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * hop >= gap * sr:
                runs.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        runs.append((start, count - quiet))
    # A click or a breath is not a sentence.
    return [(a * hop, b * hop) for a, b in runs if (b - a) * hop >= 0.5 * sr]


def import_reading(name: str, sentences: List[str], audio_path: str) -> tuple:
    """One recording of ``sentences`` read in order, a pause of about a second after each, filed as one
    take per sentence. Returns (takes saved, minutes, skipped take numbers). Nothing is saved when the pieces and the sentences
    do not line up: a take filed under the wrong text would teach the model wrong speech. Takes that line
    up but fall outside 3–15 seconds are left out; their numbers are the third value returned."""
    wav, sr = load_audio(audio_path)
    pieces = split_reading(wav, sr)
    if not pieces:
        raise ValueError("Không tìm thấy giọng nói trong bản thu.")
    if len(pieces) > len(sentences):
        raise ValueError(f"Bản thu có {len(pieces)} đoạn nói nhưng chỉ còn {len(sentences)} câu trong danh sách. "
                         "Có thể bạn đã ngừng giữa câu quá lâu; hãy ngừng ngắn hơn trong câu và khoảng một giây giữa các câu.")
    pairs = list(zip(sentences, pieces))
    # Each Vietnamese syllable takes roughly the same time, so the speaking rate must stay steady from
    # one take to the next; a take holding two sentences, or half of one, falls far off it.
    rates = [len(text.split()) / ((b - a) / sr) for text, (a, b) in pairs]
    median = float(np.median(rates))
    for n, ((text, (a, b)), rate) in enumerate(zip(pairs, rates), 1):
        if not (0.7 * median <= rate <= 1.45 * median):
            raise ValueError(f"Đoạn {n} (giây {a / sr:.0f}–{b / sr:.0f}, dài {(b - a) / sr:.1f} giây) không khớp với câu “{text[:60]}…”. "
                             "Chưa lưu gì. Hãy đọc đúng thứ tự, ngừng khoảng một giây sau mỗi câu, hoặc thu lại từ câu đó.")
    total, saved, skipped = 0.0, 0, []
    for n, (text, (a, b)) in enumerate(pairs, 1):
        try:
            total += save_take(name, text, audio=(wav[max(0, a - sr // 10):b + sr // 5], sr))
            saved += 1
        except ValueError:
            skipped.append(n)   # in line with its sentence, but too short or too long to train on
    return saved, total / 60, skipped


def passage_sentences(text: str) -> List[str]:
    """The sentences of a passage, split at sentence ends only (no joining: each was read with a pause after it)."""
    return [s.strip() for s in re.split(r"(?<=[.!?…])[\"')\]»”’]*\s+", " ".join(text.split())) if s.strip()]


def import_dataset(path, name: Optional[str] = None) -> dict:
    """Adds the recordings of a ``.dtdata`` file from the phone app (or a zip of the same layout).

    ``name`` overrides the voice name stored in the file. Sentence takes are filed as they are; a "reading"
    (the clone recorder's passage) is cut at its pauses like a long upload. Returns ``name``, ``saved``,
    ``minutes`` and ``problems`` (one line per take left out, with the reason).
    """
    import tempfile
    import zipfile
    try:
        archive = zipfile.ZipFile(str(path))
    except zipfile.BadZipFile as e:
        raise ValueError("Đây không phải file .dtdata của app điện thoại.") from e
    with archive, tempfile.TemporaryDirectory(prefix="vieneu-dtdata-") as work:
        try:
            data = json.loads(archive.read("dataset.json").decode("utf-8"))
            assert data.get("format") == "doc-truyen-dataset" and int(data.get("version", 0)) == 1
        except Exception as e:  # noqa: BLE001
            raise ValueError("Đây không phải file .dtdata của app điện thoại.") from e
        voice = safe_name(name or data.get("voice") or "")
        saved, seconds, problems = 0, 0.0, []
        for n, take in enumerate(data.get("takes", []), 1):
            file, text, kind = Path(str(take.get("file", ""))).name, str(take.get("text", "")), take.get("kind", "sentence")
            if not file or file not in archive.namelist():
                problems.append(f"bản {n}: thiếu file âm thanh"); continue
            target = Path(work) / file
            target.write_bytes(archive.read(file))
            try:
                if kind == "reading":
                    count, minutes, skipped = import_reading(voice, passage_sentences(text), str(target))
                    saved += count; seconds += minutes * 60
                    if skipped:
                        problems.append(f"bản {n} (đoạn văn): bỏ qua câu {', '.join(map(str, skipped))} vì ngắn hơn 3 hoặc dài hơn 15 giây")
                else:
                    seconds += save_take(voice, text, str(target)); saved += 1
            except ValueError as e:
                problems.append(f"bản {n}: {e}")
        return {"name": voice, "saved": saved, "minutes": seconds / 60, "problems": problems}


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
    # The voice for the phone app, in a process of its own so the card is free again. A failure here
    # does not undo the training: the same command can be run again later.
    phone = [sys.executable, "-m", "apps.phone_voice", safe_name(name)]
    print(f"\n== {' '.join(phone[1:])}", flush=True)
    if subprocess.run(phone, cwd=str(ROOT)).returncode != 0:
        print(f"⚠️ Chưa tạo được file giọng cho điện thoại. Thử lại sau bằng: .venv\\Scripts\\python.exe -m apps.phone_voice \"{safe_name(name)}\"", flush=True)
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
