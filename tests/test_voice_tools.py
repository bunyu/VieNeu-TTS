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
