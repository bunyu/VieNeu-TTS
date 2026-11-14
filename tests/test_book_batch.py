import threading
import zipfile

import numpy as np
import soundfile as sf

from apps.book_batch import collect_parts, convert_book, natural_key, pieces, read_text_file


def make_epub(path, chapters):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   "</rootfiles></container>")
        items = "".join(f'<item id="c{i}" href="Text/c{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(len(chapters)))
        # Spine order differs from manifest order on purpose.
        spine = "".join(f'<itemref idref="c{i}"/>' for i in reversed(range(len(chapters))))
        z.writestr("OEBPS/content.opf",
                   '<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f"<metadata><dc:title>Sách thử</dc:title></metadata><manifest>{items}</manifest><spine>{spine}</spine></package>")
        for i, (title, body) in enumerate(chapters):
            z.writestr(f"OEBPS/Text/c{i}.xhtml",
                       f"<html><head><title>x</title><style>p{{}}</style></head><body><h1>{title}</h1><p>{body}</p></body></html>")


class FakeTts:
    sample_rate = 8000

    def __init__(self):
        self.calls = []

    def get_preset_voice(self, name):
        return {"name": name}

    def infer(self, text, **kwargs):
        self.calls.append(text)
        return np.full(800, 0.1, dtype=np.float32)


def test_epub_chapters_follow_the_spine(tmp_path):
    book = tmp_path / "book.epub"
    make_epub(book, [("Mở đầu", "Ngày xửa ngày xưa."), ("Kết", "Hết chuyện.")])
    parts = collect_parts(book)
    assert [p.title for p in parts] == ["Kết", "Mở đầu"]
    assert parts[0].name == "001 - Kết"
    assert "Hết chuyện." in parts[0].text and "p{}" not in parts[0].text


def test_folder_parts_sort_naturally_and_mix_formats(tmp_path):
    (tmp_path / "Phan 10.txt").write_text("mười", encoding="utf-8")
    (tmp_path / "Phan 2.txt").write_bytes("hai".encode("utf-16"))
    make_epub(tmp_path / "Phan 1.epub", [("A", "một")])
    (tmp_path / "notes.md").write_text("bỏ qua", encoding="utf-8")
    parts = collect_parts(tmp_path)
    assert [p.name for p in parts] == ["Phan 1", "Phan 2", "Phan 10"]
    assert parts[1].text == "hai"


def test_natural_key_orders_numbers():
    assert natural_key("a2") < natural_key("a10")


def test_vietnamese_windows_encoding(tmp_path):
    f = tmp_path / "x.txt"
    # Windows-1258 stores most tones as combining marks; the reader recomposes them.
    # "Tiếng Việt" = ê (0xEA) + combining acute (0xEC) / combining dot below (0xF2).
    f.write_bytes(b"Ti\xea\xecng Vi\xea\xf2t")
    assert read_text_file(f) == "Tiếng Việt"


def test_pieces_keep_paragraphs_together():
    text = "\n".join(["a" * 600] * 5)
    out = list(pieces(text, limit=1500))
    assert [len(p) for p in out] == [1201, 1201, 600]


def test_resume_skips_finished_parts(tmp_path):
    src = tmp_path / "book"
    src.mkdir()
    (src / "1.txt").write_text("một", encoding="utf-8")
    (src / "2.txt").write_text("hai", encoding="utf-8")
    out = tmp_path / "out"
    tts = FakeTts()
    files = convert_book(tts, src, out, voice="Bình", fmt="wav", report=lambda _: None)
    assert [f.name for f in files] == ["1.wav", "2.wav"]
    assert sf.info(str(files[0])).samplerate == 8000
    tts2 = FakeTts()
    convert_book(tts2, src, out, fmt="wav", report=lambda _: None)
    assert tts2.calls == []


def test_stop_leaves_no_partial_file(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("một", encoding="utf-8")
    stop = threading.Event()
    stop.set()
    convert_book(FakeTts(), src, tmp_path / "out", fmt="wav", stop=stop, report=lambda _: None)
    assert list((tmp_path / "out").iterdir()) == []
