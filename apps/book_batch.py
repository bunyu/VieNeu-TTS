"""Turn a book into audio files: one file per chapter of an EPUB, or one per part in a folder.

Input is either
  * one ``.epub``  → one audio file per chapter (spine order, empty chapters skipped), or
  * one ``.txt``   → one audio file, or
  * a folder       → one audio file per ``.txt`` / ``.epub`` part, sorted naturally by name
                     ("Phan 2" before "Phan 10").

Each part is written to ``<output>/<NNN - name>.<format>``, or several parts to one file with
``chapters_per_file`` (``001-005 - name``). A file that is already complete is skipped, so an
interrupted run resumes where it stopped; files are written under a temporary name and renamed
only when finished.

Pauses: ``paragraph_pause`` between paragraphs, ``[n]`` tags for an exact stop, and
``pause_rules`` (mark -> seconds) to stop for exactly that long after a punctuation mark, as in
the Android reader. A mark without a rule keeps the model's own flow.

Next to every audio file goes ``<same name>.sync.json``: the text of each paragraph with the second
it starts and ends at. The Android reader imports the audio files together with these, and then
shows the text and follows the audio in it.

Command line (uses v3 Turbo on the GPU when one is available):
    python -m apps.book_batch BOOK_OR_FOLDER [--out DIR] [--voice NAME] [--format mp3|wav|flac]
        [--chapters-per-file N] [--paragraph-pause S] [--pause ".=0.6" --pause ",=0.2" ...]
"""
from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np
import soundfile as sf

# "opus" is the one to carry to the phone: small, and a player can jump to an exact time in it.
FORMATS = {"mp3": ("MP3", "MPEG_LAYER_III"), "wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"), "opus": ("OGG", "OPUS")}
SYNC_SUFFIX = ".sync.json"
PIECE_CHARS = 1500  # text handed to the engine per call; bounds memory and gives progress ticks
PAUSE_TAG = re.compile(r"\[\s*(\d{1,2}(?:[.,]\d{1,3})?)\s*\]")
MAX_TAG_SECONDS = 10.0
# Marks that can be given an exact pause, with the pause the model makes by itself (shown as the default).
PAUSE_MARKS = {".": 0.5, ",": 0.3, "?": 0.5, "!": 0.5, ":": 0.4, ";": 0.4, "…": 0.8}
CHAPTER_PAUSE = 1.4  # seconds between chapters that share a file
NO_BREAK = {"tp", "gs", "pgs", "ts", "ths", "bs", "ks", "v.v", "mr", "mrs", "ms", "dr", "st", "tr", "q", "p", "ông", "bà", "no", "vol"}
CLOSING = "\"')]»”’"


@dataclass
class Part:
    name: str   # used for the output file name
    title: str  # shown in progress
    text: str


# ── Reading books ────────────────────────────────────────────────────────────

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", name)]


def read_text_file(path: Path) -> str:
    """UTF-8/UTF-16 (BOM) or, failing UTF-8, Windows-1258 Vietnamese; always NFC."""
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        text = data[3:].decode("utf-8")
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp1258", errors="replace")
    return unicodedata.normalize("NFC", text)


class _HtmlText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote", "section", "article"}
    SKIP = {"script", "style", "head", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.heading: Optional[str] = None
        self._skip = 0
        self._in_heading = False
        self._heading_buf: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3") and self.heading is None:
            self._in_heading = True

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3") and self._in_heading:
            self._in_heading = False
            self.heading = " ".join("".join(self._heading_buf).split()) or None

    def handle_data(self, data):
        if self._skip:
            return
        self.out.append(data)
        if self._in_heading:
            self._heading_buf.append(data)


def html_to_text(markup: str) -> tuple[str, Optional[str]]:
    parser = _HtmlText()
    parser.feed(markup)
    lines = [" ".join(line.split()) for line in "".join(parser.out).split("\n")]
    return "\n".join(line for line in lines if line), parser.heading


def _xml(data: bytes) -> ET.Element:
    return ET.fromstring(data)


def read_epub(path: Path) -> tuple[str, List[tuple[str, str]]]:
    """Book title and its chapters as (title, text), in reading (spine) order."""
    with zipfile.ZipFile(path) as book:
        container = _xml(book.read("META-INF/container.xml"))
        rootfile = next(el for el in container.iter() if el.tag.endswith("rootfile")).attrib["full-path"]
        opf = _xml(book.read(rootfile))
        base = posixpath.dirname(rootfile)
        title_el = next((el for el in opf.iter() if el.tag.endswith("}title") and (el.text or "").strip()), None)
        book_title = title_el.text.strip() if title_el is not None else path.stem
        manifest = {el.attrib["id"]: el.attrib for el in opf.iter() if el.tag.endswith("}item") and "id" in el.attrib}
        names = set(book.namelist())
        chapters = []
        for ref in (el for el in opf.iter() if el.tag.endswith("}itemref")):
            item = manifest.get(ref.attrib.get("idref", ""))
            if not item or "html" not in item.get("media-type", "html"):
                continue
            href = posixpath.normpath(posixpath.join(base, html.unescape(item["href"]).split("#")[0]))
            if href not in names:
                continue
            text, heading = html_to_text(book.read(href).decode("utf-8", errors="replace"))
            if len(text) < 2:
                continue
            chapters.append((heading or f"Chương {len(chapters) + 1}", text))
    return book_title, chapters


def collect_parts(source: Path) -> List[Part]:
    source = Path(source)
    if source.is_dir():
        files = sorted((p for p in source.iterdir() if p.suffix.lower() in (".txt", ".epub") and p.is_file()),
                       key=lambda p: natural_key(p.name))
        parts = []
        for f in files:
            if f.suffix.lower() == ".txt":
                text = read_text_file(f)
            else:
                text = "\n\n".join(t for _, t in read_epub(f)[1])
            if text.strip():
                parts.append(Part(f.stem, f.name, text))
        return parts
    if source.suffix.lower() == ".epub":
        _, chapters = read_epub(source)
        width = max(3, len(str(len(chapters))))
        return [Part(f"{i:0{width}d} - {title}", title, text) for i, (title, text) in enumerate(chapters, 1)]
    if source.suffix.lower() == ".txt":
        return [Part(source.stem, source.name, read_text_file(source))]
    raise ValueError(f"Không đọc được {source}: chỉ hỗ trợ .epub, .txt hoặc thư mục chứa chúng.")


def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    return " ".join(name.split()).strip(" .")[:120] or "part"


def default_output_dir(source: Path) -> Path:
    source = Path(source).resolve()
    return source.parent / f"{source.stem if source.is_file() else source.name} - audio"


# ── Synthesis ────────────────────────────────────────────────────────────────

def pieces(text: str, limit: int = PIECE_CHARS) -> Iterator[str]:
    """Paragraph-aligned slices of at most ``limit`` characters (a longer paragraph is its own slice)."""
    buf: List[str] = []
    size = 0
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if buf and size + len(para) > limit:
            yield "\n".join(buf)
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 1
    if buf:
        yield "\n".join(buf)


def split_pause_tags(text: str) -> List[tuple]:
    """``[0.8]`` / ``[2]`` tags (seconds, at most 10 each, ``[10][10]`` = 20 s) → [(text, pause_after_seconds)]."""
    out: List[tuple] = []
    at = 0
    pending = 0.0
    for m in PAUSE_TAG.finditer(text):
        chunk = text[at:m.start()]
        if chunk.strip():
            out.append([chunk.strip(), 0.0])
        seconds = min(float(m.group(1).replace(",", ".")), MAX_TAG_SECONDS)
        if out and not chunk.strip():
            out[-1][1] += seconds
        elif out:
            out[-1][1] = seconds
        else:
            out.append(["", seconds])
        at = m.end()
    rest = text[at:]
    if rest.strip():
        out.append([rest.strip(), 0.0])
    return [tuple(x) for x in out]


def split_marks(text: str, rules: dict) -> List[tuple]:
    """Cut one paragraph after every mark that has a rule → [(text, seconds or None)].

    The last stretch, and any stretch ending in a mark without a rule, has None. A mark only ends
    a stretch when a space or the end follows (not "3.5", "a,b" or "Tp.HCM"), and never after an
    abbreviation or an initial.
    """
    if not rules:
        return [(text.strip(), None)] if text.strip() else []
    out: List[tuple] = []
    start = 0
    i = 0
    while i < len(text):
        length = 3 if text.startswith("...", i) else 1
        mark = "…" if length == 3 else text[i]
        if mark in rules and _breaks_here(text, i, length, mark):
            j = i + length
            while j < len(text) and text[j] in CLOSING:
                j += 1
            chunk = text[start:j].strip()
            if out and not any(ch.isalnum() for ch in chunk):
                out[-1] = (out[-1][0] + chunk, (out[-1][1] or 0.0) + rules[mark])  # "?!" or a stray mark: one stop
            elif chunk:
                out.append((chunk, rules[mark]))
            start = i = j
            continue
        i += length
    rest = text[start:].strip()
    if rest:
        out.append((rest, None))
    return out


def _breaks_here(text: str, at: int, length: int, mark: str) -> bool:
    j = at + length
    while j < len(text) and text[j] in CLOSING:
        j += 1
    if j < len(text) and not text[j].isspace():
        return False
    if mark == ".":
        if (at > 0 and text[at - 1] == ".") or text.startswith("..", at):
            return False  # the dots of "..." are one ellipsis
        word = text[:at].split()[-1].lower() if text[:at].split() else ""
        if word in NO_BREAK or (len(word) == 1 and word.isalpha()):
            return False
    return True


def trim_tail(wav: np.ndarray, sample_rate: int, keep: float = 0.04) -> np.ndarray:
    """Drop the quiet end of a clip (leaving ``keep`` seconds) so an exact pause is not added to the model's own."""
    window = max(1, sample_rate // 100)
    end = len(wav)
    while end > 0 and np.max(np.abs(wav[max(0, end - window):end])) < 0.012:
        end -= window
    return wav[:min(len(wav), end + int(keep * sample_rate))]


def group_parts(parts: List[Part], per_file: int) -> List[tuple]:
    """Parts gathered ``per_file`` at a time → [(file name, title, parts)]."""
    if per_file <= 1:
        return [(p.name, p.title, [p]) for p in parts]
    width = max(3, len(str(len(parts))))
    groups = []
    for first in range(0, len(parts), per_file):
        group = parts[first:first + per_file]
        if len(group) == 1:
            groups.append((group[0].name, group[0].title, group))
            continue
        span = f"{first + 1:0{width}d}-{first + len(group):0{width}d}"
        # The first part's own name, without the number an EPUB chapter carries ("001 - Mở đầu").
        groups.append((f"{span} - {re.sub(r'^[0-9]+ - ', '', group[0].name)}", f"{group[0].title} … {group[-1].title}", group))
    return groups


def _speak(tts, texts: List[str], kwargs: dict, batch: int = 8) -> List[np.ndarray]:
    """One clip per text; batched on engines that can, a few at a time to stay inside a small GPU."""
    if len(texts) > 1 and hasattr(tts, "infer_batch"):
        out: List[np.ndarray] = []
        for at in range(0, len(texts), batch):
            out += [np.asarray(w, dtype=np.float32).reshape(-1) for w in tts.infer_batch(texts[at:at + batch], **kwargs)]
        return out
    return [np.asarray(tts.infer(t, **kwargs), dtype=np.float32).reshape(-1) for t in texts]


def render_piece(tts, piece: str, kwargs: dict, paragraph_pause: float, rules: dict) -> tuple:
    """One slice of text → (audio, marks). ``marks`` holds (paragraph, first sample, end sample) for each
    paragraph, its closing pause included, so a reader can follow the audio in the text."""
    rate = tts.sample_rate
    audio: List[np.ndarray] = []
    marks: List[tuple] = []
    size = 0

    def add(wav):
        nonlocal size
        audio.append(wav)
        size += len(wav)

    for spoken, tag_pause in split_pause_tags(piece):
        paragraphs = [p.strip() for p in spoken.split("\n") if any(ch.isalnum() for ch in p)]
        # Every paragraph is spoken by itself; one with a rule on a mark is first cut after those marks.
        units: List[tuple] = []  # (text, pause after it or None, paragraph, last of its paragraph)
        for n, paragraph in enumerate(paragraphs):
            stretches = [s for s in split_marks(paragraph, rules) if any(ch.isalnum() for ch in s[0])]
            for k, (text, pause) in enumerate(stretches):
                last = k == len(stretches) - 1
                # A paragraph end takes the paragraph pause; a rule on its last mark can only lengthen it.
                units.append((text, max(paragraph_pause, pause or 0.0) if last else pause, n, last))
        if not units:
            add(np.zeros(int((tag_pause if tag_pause > 0 else paragraph_pause) * rate), np.float32))
            continue
        if tag_pause > 0:
            units[-1] = (units[-1][0], tag_pause) + units[-1][2:]
        start = size
        for (text, pause, n, last), wav in zip(units, _speak(tts, [u[0] for u in units], kwargs)):
            add(trim_tail(wav, rate) if rules and pause is not None else wav)
            if pause is not None:
                add(np.zeros(int(pause * rate), np.float32))
            if last:
                marks.append((paragraphs[n], start, size))
                start = size
    return (np.concatenate(audio) if audio else np.zeros(0, np.float32)), marks


def book_title(source: Path) -> str:
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".epub":
        try:
            return read_epub(source)[0]
        except Exception:  # noqa: BLE001
            pass
    return source.stem if source.is_file() else source.name


def convert_book(tts, source, out_dir=None, voice: Optional[str] = None, fmt: str = "mp3",
                 stop: Optional[threading.Event] = None,
                 report: Callable[[str], None] = print, paragraph_pause: float = 0.35,
                 chapters_per_file: int = 1, pause_rules: Optional[dict] = None) -> List[Path]:
    """Write the book as audio files; returns the files written or already present.

    ``chapters_per_file``: how many parts (EPUB chapters) go into one file. ``paragraph_pause``:
    seconds of silence between paragraphs. ``pause_rules``: mark → exact seconds (see
    ``PAUSE_MARKS``). ``[n]`` tags in the text add exact pauses.
    """
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {sorted(FORMATS)}")
    parts = collect_parts(Path(source))
    if not parts:
        raise ValueError("Không tìm thấy nội dung (.txt / .epub) để đọc.")
    rules = {m: min(max(float(s), 0.0), MAX_TAG_SECONDS) for m, s in (pause_rules or {}).items() if m in PAUSE_MARKS}
    groups = group_parts(parts, max(1, int(chapters_per_file)))
    out = Path(out_dir) if out_dir else default_output_dir(Path(source))
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {"voice": tts.get_preset_voice(voice)} if voice else {}
    container, subtype = FORMATS[fmt]
    title_of_book = book_title(Path(source))
    total_chars = sum(len(p.text) for p in parts)
    done_chars = 0
    started = time.perf_counter()
    written: List[Path] = []
    stopped = lambda: stop is not None and stop.is_set()  # noqa: E731
    for index, (name, title, group) in enumerate(groups, 1):
        target = out / f"{safe_name(name)}.{fmt}"
        sync = target.with_name(target.name + SYNC_SUFFIX)
        if target.exists() and target.stat().st_size > 0:
            # Audio made before sync files existed is kept as it is; delete the file to have it made again with one.
            report(f"[{index}/{len(groups)}] ✔ đã có, bỏ qua: {target.name}" + ("" if sync.exists() else " (chưa có file .sync.json)"))
            done_chars += sum(len(p.text) for p in group)
            written.append(target)
            continue
        partial = target.with_name(target.name + ".partial")
        report(f"[{index}/{len(groups)}] ▶ {title}")
        file_audio_s = 0.0
        samples = 0
        chapters: List[dict] = []
        with sf.SoundFile(str(partial), "w", samplerate=tts.sample_rate, channels=1,
                          format=container, subtype=subtype) as sink:
            for at, part in enumerate(group):
                if at:
                    gap = np.zeros(int(CHAPTER_PAUSE * tts.sample_rate), np.float32)
                    sink.write(gap)
                    samples += len(gap)
                    report(f"    ▶ {part.title}")
                segments: List[list] = []
                chapters.append({"title": part.title, "segments": segments})
                for piece in pieces(part.text):
                    if stopped():
                        break
                    wav, marks = render_piece(tts, piece, kwargs, paragraph_pause, rules)
                    sink.write(wav)
                    segments += [[round((samples + a) / tts.sample_rate, 3), round((samples + b) / tts.sample_rate, 3), text] for text, a, b in marks]
                    samples += len(wav)
                    file_audio_s = samples / tts.sample_rate
                    done_chars += len(piece)
                    elapsed = time.perf_counter() - started
                    eta = elapsed / max(done_chars, 1) * (total_chars - done_chars)
                    report(f"    {done_chars * 100 // total_chars}% · {file_audio_s / 60:.1f} phút audio · còn ~{eta / 60:.0f} phút")
                if stopped():
                    break
        if stopped():
            partial.unlink(missing_ok=True)
            report("⏹️ Đã dừng. Chạy lại để tiếp tục từ phần này.")
            return written
        sync.write_text(json.dumps({"format": "vieneu-sync", "version": 1, "book": title_of_book, "index": index, "of": len(groups),
                                    "audio": target.name, "seconds": round(samples / tts.sample_rate, 3), "chapters": chapters},
                                   ensure_ascii=False), encoding="utf-8")
        partial.replace(target)
        written.append(target)
        report(f"    ✅ {target.name} ({file_audio_s / 60:.1f} phút)")
    report(f"🎉 Xong {len(written)}/{len(groups)} file trong {(time.perf_counter() - started) / 60:.1f} phút → {out}")
    return written


def parse_pause_rules(items: Optional[List[str]]) -> dict:
    """``[".=0.6", ",=0.2"]`` → {".": 0.6, ",": 0.2}; "..." stands for the ellipsis."""
    rules = {}
    for item in items or []:
        mark, _, seconds = item.rpartition("=")
        mark = "…" if mark.strip() == "..." else mark.strip()
        if mark not in PAUSE_MARKS:
            raise ValueError(f"--pause: mark must be one of {' '.join(PAUSE_MARKS)} (got {item!r})")
        rules[mark] = float(seconds.replace(",", "."))
    return rules


def load_turbo():
    """v3 Turbo on this PC's GPU when there is one, otherwise ONNX on the CPU."""
    from vieneu import Vieneu
    from vieneu.gpu_support import turbo_gpu_kwargs
    kwargs = turbo_gpu_kwargs() or {"backend": "onnx"}
    tts = Vieneu("v3turbo", **kwargs)
    try:
        from apps.user_voices import load_user_voices
        load_user_voices(tts)  # voices saved in the web app's Voice Cloning tab
    except Exception:
        pass
    return tts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="an .epub, a .txt, or a folder of .txt/.epub parts")
    parser.add_argument("--out", help="output folder (default: '<source> - audio' next to it)")
    parser.add_argument("--voice", help="preset or saved voice name (default: the model's default voice)")
    parser.add_argument("--format", default="mp3", choices=sorted(FORMATS))
    parser.add_argument("--list", action="store_true", help="only list the parts that would be read")
    parser.add_argument("--paragraph-pause", type=float, default=0.35, help="seconds between paragraphs")
    parser.add_argument("--chapters-per-file", type=int, default=1, help="chapters (parts) put into one audio file")
    parser.add_argument("--pause", action="append", metavar="MARK=SECONDS",
                        help="exact pause after a punctuation mark, e.g. --pause \".=0.6\" (repeatable)")
    args = parser.parse_args()
    if args.list:
        for i, p in enumerate(collect_parts(Path(args.source)), 1):
            print(f"{i:4d}. {p.title}  ({len(p.text)} ký tự)")
        return
    convert_book(load_turbo(), args.source, args.out, args.voice, args.format, paragraph_pause=args.paragraph_pause,
                 chapters_per_file=args.chapters_per_file, pause_rules=parse_pause_rules(args.pause))


if __name__ == "__main__":
    main()
