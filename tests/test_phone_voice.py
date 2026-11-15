import json
import zipfile

import numpy as np

from apps import phone_voice


class FakeTurbo:
    sample_rate = 48000

    def __init__(self):
        self.voices = []

    def get_preset_voice(self, name):
        return {"name": name}

    def infer(self, text, voice=None):
        self.voices.append(voice)
        return np.full(48000 * 6, 0.1, np.float32)


class FakeNano:
    def __init__(self):
        self.enrolled = None

    def prepare_reference(self, path, denoise=True):
        self.enrolled = (path, denoise)
        return np.linspace(-1, 1, 192, dtype=np.float32), np.ones((50, 256), np.float32) * 0.5

    def infer(self, phonemes, speaker, style, **kwargs):
        return np.zeros(24000, np.float32)


def test_export_writes_a_phone_voice(tmp_path):
    turbo, nano = FakeTurbo(), FakeNano()
    result = phone_voice.export(turbo, "Giọng tôi", "Giọng: bố/mẹ", tmp_path, description="trầm", nano=nano, report=lambda _: None)
    assert turbo.voices == [{"name": "Giọng tôi"}]
    assert nano.enrolled[1] is False                      # clean speech is not denoised
    with zipfile.ZipFile(result["file"]) as z:
        assert sorted(z.namelist()) == ["sample.wav", "voice.json"]
        data = json.loads(z.read("voice.json"))
    assert result["file"].endswith("Giọng bố mẹ.dtvoice")
    assert (data["format"], data["version"], data["name"], data["description"]) == ("doc-truyen-voice", 1, "Giọng: bố/mẹ", "trầm")
    assert data["release"] == phone_voice.release()["id"]
    assert len(data["speaker"]) == 192 and len(data["style"]) == 50 and all(len(r) == 256 for r in data["style"])
    assert data["speaker"][0] == -1.0 and data["style"][3][7] == 0.5


def test_the_phone_release_lists_every_file_nano_needs():
    names = {f["name"] for f in phone_voice.release()["files"]}
    assert set(phone_voice.NEEDED) <= names
