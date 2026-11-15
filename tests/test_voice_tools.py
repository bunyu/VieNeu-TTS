import numpy as np
import pytest
import soundfile as sf

from apps import finetune_studio as ft
from apps.voice_lab import candidate_windows, cosine, voiced_frames


def speech(seconds, sr=16000, level=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_candidates_are_full_of_speech_and_do_not_overlap():
    sr = 16000
    quiet = np.zeros(4 * sr, np.float32)
    wav = np.concatenate([quiet, speech(10), quiet, speech(9), quiet, speech(2), quiet])
    windows = candidate_windows(wav, sr, seconds=8.0, count=4)
    assert len(windows) == 2                      # the two-second burst cannot fill a window
    assert all(b - a == 8 * sr for a, b in windows)
    assert windows[0][1] <= windows[1][0]
    for a, b in windows:
        voiced, _ = voiced_frames(wav[a:b], sr)
        assert voiced.mean() > 0.95
    # A phrase start is preferred to the middle of speech: each window begins just before its burst.
    assert abs(windows[0][0] - 4 * sr) <= 0.1 * sr
    assert candidate_windows(speech(5), sr) == [(0, 5 * sr)]
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0) and cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def test_sentences_for_recording_are_one_breath_long():
    text = "Ừ. Thế à? Anh không biết chuyện ấy đâu, thật đấy. " + "Một câu rất dài " * 20 + ". Trời đã tối hẳn khi anh về đến nhà, cô vẫn ngồi chờ bên cửa sổ."
    sentences = ft.split_sentences(text)
    assert sentences[0] == "Ừ. Thế à? Anh không biết chuyện ấy đâu, thật đấy."
    assert sentences[-1].startswith("Trời đã tối hẳn")
    assert all(35 <= len(s) <= 170 for s in sentences)
    assert len(ft.load_script()) == 40 and all(35 <= len(s) <= 170 for s in ft.load_script())


def test_takes_are_filed_with_their_text(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "DATASETS", tmp_path)
    sr = 16000
    take = tmp_path / "take.wav"
    sf.write(take, np.concatenate([np.zeros(sr, np.float32), speech(4), np.zeros(sr, np.float32)]), sr)
    seconds = ft.save_take("Giong toi", "Câu  thứ nhất.", str(take))
    assert 4.0 <= seconds <= 4.6                  # the silence around the sentence is cut
    ft.save_take("Giong toi", "Câu thứ hai.", str(take))
    ft.save_take("Giong toi", "Câu thứ nhất.", str(take))   # a second take replaces the first
    folder = tmp_path / "Giong toi"
    assert (folder / "metadata.csv").read_text(encoding="utf-8") == "0001.wav|Câu thứ nhất.\n0002.wav|Câu thứ hai.\n"
    assert sorted(p.name for p in (folder / "raw_audio").iterdir()) == ["0001.wav", "0002.wav"]
    count, minutes, done = ft.progress("Giong toi")
    assert count == 2 and done == {"Câu thứ nhất.", "Câu thứ hai."} and 0.13 < minutes < 0.16
    short = tmp_path / "short.wav"
    sf.write(short, speech(1.5), sr)
    with pytest.raises(ValueError):
        ft.save_take("Giong toi", "Câu quá ngắn.", str(short))
    with pytest.raises(ValueError):
        ft.safe_name("  ")


def test_a_long_reading_is_cut_into_its_sentences(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "DATASETS", tmp_path)
    sr = 16000
    rng = np.random.default_rng(1)
    noise = lambda seconds: (0.003 * rng.normal(size=int(seconds * sr))).astype(np.float32)   # a phone's noise floor
    sentences = ["Một hai ba bốn năm sáu bảy tám chín mười.", "Mười một mười hai mười ba mười bốn mười lăm mười sáu.",
                 "Hai mươi mốt hai mươi hai hai mươi ba.", "Câu này không được đọc."]
    # About 3 syllables a second; a short comma pause inside the second sentence must not split it.
    reading = [noise(1), speech(3.4), noise(1.1), speech(2.0), noise(0.3), speech(2.0), noise(1.0), speech(2.3), noise(1.5)]
    take = tmp_path / "phone.wav"
    sf.write(take, np.concatenate(reading) + noise(sum(len(r) for r in reading) / sr), sr)
    count, minutes, skipped = ft.import_reading("Giong toi", sentences, str(take))
    # The third sentence lines up but lasts under three seconds: left out and reported, not a failure.
    assert (count, skipped) == (2, [3]) and 0.12 < minutes < 0.16
    assert ft.progress("Giong toi")[2] == set(sentences[:2])
    # Forgetting the pause between two sentences joins them: the rate is off, and nothing is saved.
    # (Sentences two and three read as one stretch, then sentence four taken for sentence three.)
    sf.write(take, np.concatenate([noise(1), speech(3.4), noise(1.1), speech(6.4), noise(1), speech(1.7), noise(1)]), sr)
    with pytest.raises(ValueError, match="không khớp"):
        ft.import_reading("Giong khac", sentences, str(take))
    assert ft.progress("Giong khac")[0] == 0


def test_phone_formats_are_read(tmp_path):
    av = pytest.importorskip("av")
    sr = 48000
    path = tmp_path / "phone.m4a"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("aac", rate=sr, layout="mono")
        frame = av.AudioFrame.from_ndarray(speech(2, sr)[None, :], format="flt", layout="mono")
        frame.sample_rate = sr
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    wav, rate = ft.load_audio(path)
    assert rate == sr and 1.9 < len(wav) / sr < 2.2 and np.abs(wav).max() > 0.1


def test_the_phone_and_the_pc_offer_the_same_sentences():
    assert [l.strip() for l in ft.PHONE_SENTENCES.read_text(encoding="utf-8").splitlines() if l.strip()] == ft.load_script()


def test_recordings_from_the_phone_are_added(tmp_path, monkeypatch):
    import json, zipfile
    monkeypatch.setattr(ft, "DATASETS", tmp_path / "datasets")
    sr = 48000
    rng = np.random.default_rng(2)
    noise = lambda seconds: (0.003 * rng.normal(size=int(seconds * sr))).astype(np.float32)
    def wav(parts):
        path = tmp_path / f"w{len(list(tmp_path.glob('w*')))}.wav"
        sf.write(path, np.concatenate(parts), sr, subtype="PCM_16")
        return path
    passage = "Hôm nay trời đẹp, tôi ngồi bên cửa sổ. Ngoài vườn chim hót líu lo trên cành cây cao. Bà tưới rau."
    files = {
        "0001.wav": wav([noise(0.3), speech(4, sr), noise(0.3)]),
        "0002.wav": wav([noise(0.3), speech(1.5, sr), noise(0.3)]),                         # too short to train on
        "0003.wav": wav([noise(0.5), speech(3.0, sr), noise(0.9), speech(4.0, sr), noise(0.9), speech(1.3, sr), noise(0.5)]),
    }
    dataset = {"format": "doc-truyen-dataset", "version": 1, "voice": "Giọng bố", "takes": [
        {"file": "0001.wav", "text": "Câu thứ nhất của bố.", "kind": "sentence"},
        {"file": "0002.wav", "text": "Câu ngắn.", "kind": "sentence"},
        {"file": "0003.wav", "text": passage, "kind": "reading"},
        {"file": "0009.wav", "text": "Mất file.", "kind": "sentence"}]}
    package = tmp_path / "Giong bo.dtdata"
    with zipfile.ZipFile(package, "w") as z:
        z.writestr("dataset.json", json.dumps(dataset, ensure_ascii=False))
        for name, path in files.items():
            z.write(path, name)
    result = ft.import_dataset(package)
    assert result["name"] == "Giọng bố" and result["saved"] == 3        # one sentence and two of the passage's three
    assert len(result["problems"]) == 3                                   # the short take, the passage's short sentence, the missing file
    assert ft.progress("Giọng bố")[2] == {"Câu thứ nhất của bố.", "Hôm nay trời đẹp, tôi ngồi bên cửa sổ.", "Ngoài vườn chim hót líu lo trên cành cây cao."}
    assert ft.import_dataset(package, "Tên khác")["name"] == "Tên khác"
    with pytest.raises(ValueError):
        ft.import_dataset(files["0001.wav"])
