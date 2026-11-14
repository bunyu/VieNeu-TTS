"""Turn a book into audio files: one file per chapter of an EPUB, or one per part in a folder.

Input is either
  * one ``.epub``  → one audio file per chapter (spine order, empty chapters skipped), or
  * one ``.txt``   → one audio file, or
  * a folder       → one audio file per ``.txt`` / ``.epub`` part, sorted naturally by name
                     ("Phan 2" before "Phan 10").

Each part is written to ``<output>/<NNN - name>.<format>``. A part that is already complete
is skipped, so an interrupted run resumes where it stopped; files are written under a
temporary name and renamed only when finished.

Command line (uses v3 Turbo on the GPU when one is available):
    python -m apps.book_batch BOOK_OR_FOLDER [--out DIR] [--voice NAME] [--format mp3|wav|flac]
"""
from __future__ import annotations

import argparse
import html
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

FORMATS = {"mp3": ("MP3", "MPEG_LAYER_III"), "wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16")}
PIECE_CHARS = 1500  # text handed to the engine per call; bounds memory and gives progress ticks
PAUSE_TAG = re.compile(r"\[\s*(\d{1,2}(?:[.,]\d{1,3})?)\s*\]")
MAX_TAG_SECONDS = 10.0


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


def convert_book(tts, source, out_dir=None, voice: Optional[str] = None, fmt: str = "mp3",
                 stop: Optional[threading.Event] = None,
                 report: Callable[[str], None] = print, paragraph_pause: float = 0.35) -> List[Path]:
    """``paragraph_pause``: seconds of silence between paragraphs; ``[n]`` tags in the text add exact pauses."""
    """Write one audio file per part; returns the files written or already present."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {sorted(FORMATS)}")
    parts = collect_parts(Path(source))
    if not parts:
        raise ValueError("Không tìm thấy nội dung (.txt / .epub) để đọc.")
    out = Path(out_dir) if out_dir else default_output_dir(Path(source))
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {"voice": tts.get_preset_voice(voice)} if voice else {}
    container, subtype = FORMATS[fmt]
    total_chars = sum(len(p.text) for p in parts)
    done_chars = 0
    started = time.perf_counter()
    written: List[Path] = []
    for index, part in enumerate(parts, 1):
        target = out / f"{safe_name(part.name)}.{fmt}"
        if target.exists() and target.stat().st_size > 0:
            report(f"[{index}/{len(parts)}] ✔ đã có, bỏ qua: {target.name}")
            done_chars += len(part.text)
            written.append(target)
            continue
        partial = target.with_name(target.name + ".partial")
        report(f"[{index}/{len(parts)}] ▶ {part.title}")
        part_audio_s = 0.0
        with sf.SoundFile(str(partial), "w", samplerate=tts.sample_rate, channels=1,
                          format=container, subtype=subtype) as sink:
            for piece in pieces(part.text):
                if stop is not None and stop.is_set():
                    break
                for spoken, pause in split_pause_tags(piece):
                    wav = (np.asarray(tts.infer(spoken, **kwargs), dtype=np.float32).reshape(-1)
                           if any(ch.isalnum() for ch in spoken) else np.zeros(0, np.float32))
                    gap = pause if pause > 0 else paragraph_pause
                    sink.write(np.concatenate([wav, np.zeros(int(gap * tts.sample_rate), np.float32)]))
                    part_audio_s += (len(wav) / tts.sample_rate) + gap
                done_chars += len(piece)
                elapsed = time.perf_counter() - started
                eta = elapsed / max(done_chars, 1) * (total_chars - done_chars)
                report(f"    {done_chars * 100 // total_chars}% · {part_audio_s / 60:.1f} phút audio · còn ~{eta / 60:.0f} phút")
        if stop is not None and stop.is_set():
            partial.unlink(missing_ok=True)
            report("⏹️ Đã dừng. Chạy lại để tiếp tục từ phần này.")
            return written
        partial.replace(target)
        written.append(target)
        report(f"    ✅ {target.name} ({part_audio_s / 60:.1f} phút)")
    report(f"🎉 Xong {len(written)}/{len(parts)} phần trong {(time.perf_counter() - started) / 60:.1f} phút → {out}")
    return written


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
    args = parser.parse_args()
    if args.list:
        for i, p in enumerate(collect_parts(Path(args.source)), 1):
            print(f"{i:4d}. {p.title}  ({len(p.text)} ký tự)")
        return
    convert_book(load_turbo(), args.source, args.out, args.voice, args.format, paragraph_pause=args.paragraph_pause)


if __name__ == "__main__":
    main()
