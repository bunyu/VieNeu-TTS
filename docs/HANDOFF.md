# Handoff: VieNeu Reader (Android) and the Windows GPU app

State as of 2026-11-14, `main` at `b9b4286` on `bunyu/VieNeu-TTS` (a fork of
`pnnbao97/VieNeu-TTS`, Apache-2.0). Start a new feature from `main` on a branch of
its own; push to `main` only once the work is finished and tested.

## What exists

| Piece | Where | State |
|---|---|---|
| Android EPUB reader (VieNeu v3 Nano on ONNX Runtime) | `android/` (Gradle project, module `:app`) | Tested by unit tests and a real-model test on the PC; not yet exercised on the phone after the latest changes |
| Windows web app with GPU (v3 Turbo on PyTorch/CUDA) | `apps/gradio_main.py`, `run_windows.bat`, `setup_windows_gpu.bat` | Verified in the browser on the GTX 1060 |
| Book to audio (EPUB per chapter, or a folder of `.txt`/`.epub` parts) | `apps/book_batch.py`, tab "📚 Sách → audio" | Verified on the GPU |
| Clone-similarity measurement | `tests/clone_similarity.py` | Used for the numbers below |

`android/README.md` describes the reader's features and design in full; read it first
when touching the phone app.

## Reader: map of the code (`android/app/src/main/kotlin/io/github/bunyu/reader/`)

- `vieneu/` the model. `NanoEngine` (sampling loop, port of upstream `vieneu/v3nano.py`),
  `NanoEnroller` (voice cloning), `NanoDenoiser` (port of upstream's resemble-enhance
  denoiser, with its own STFT/FFT), `SpeechFeatures` (fbank, resampling), `SeaG2p`
  (JNI to the Rust phonemiser in `app/native/g2p`), `NanoAssets` / `NanoRelease`
  (pinned downloads from `assets/vieneu-release.json`: URL + size + sha256, group
  `speech` or `clone`).
- `playback/` `Narration` (what plays, what is prepared ahead, "prepare whole book"),
  `Renderer` (one book piece → one cached `.m4a`; the cache key is in
  `Renderer.resolve`), `AudioPolish` (trim, loudness, pause), `ReaderService`
  (foreground service, notification).
- `text/` `Chunker` (pieces of a few sentences), `VietnameseText` (numbers, dates,
  dialect, user pronunciations), `PauseRules` / `PauseText` (pause settings and `[n]`
  tags).
- `voice/ReaderVoices` stores cloned voices (`print.f32` + original clip);
  `voice/VoiceBundle` imports VoiceStudio exports.
- `media/` decoding, AAC writing, `VoiceRecorder` (AudioRecord, 48 kHz mono).
- `ui/` Compose screens: `LibraryScreen`, `ReaderScreen`, `VoicesScreen`
  (+ `RecordDialog`), `SettingsScreen`, `PauseDialog`.
- Strings live in `res/values/strings.xml` (English) and `res/values-vi/strings.xml`
  (Vietnamese); every new string needs both.

### Behaviour worth knowing before changing it

- **Audio cache.** Each prepared piece is kept in `filesDir/books/<id>/audio/<key>.m4a`
  until the book's audio is cleared. The key hashes the model release, fast mode,
  `VietnameseText.VERSION`, dialect, pronunciations, the voice (and its `enrolledAt`),
  the changed pause rules, and the piece text and pause. Anything that changes the
  sound must join that key; anything that must not invalidate old audio must stay out
  of it (pause rules add to it only when changed).
- **Prepare whole book** starts at the current piece (`NarrationState.preparedFrom`) and
  counts progress from there.
- **Pauses.** A mark left at its default keeps the model's own flow; a changed mark is
  split there and followed by exactly that silence (`PauseText.segments`, then
  `Renderer.render`, `endPause`). `[0.8]` / `[2]` tags (≤ 10 s each, they add up) are
  not spoken. `AudioPolish` still keeps about 60 ms of breath around each part, so an
  "exact" pause is that much longer.
- **Cloning.** `NanoEnroller` denoises only when the estimated SNR is below 30 dB
  (`NanoDenoiser.NOISY_BELOW_DB`), and takes style tokens from the most voiced 5 s.
  Measured with `tests/clone_similarity.py` on the 5 clips in `examples/audio_ref`
  (similarity of cloned speech to the speaker):

  | Reference | No denoise | Denoise | Denoise only when noisy (shipped) |
  |---|---|---|---|
  | clean | 0.779 | 0.760 | 0.773 |
  | room noise, SNR 25 dB | 0.711 | 0.721 | 0.721 |
  | room noise, SNR 15 dB | 0.659 | 0.705 | 0.705 |

  Guidance 4 instead of 3, 8 vs 16 steps and the voiced-window choice moved this by
  ±0.01. The Kotlin denoiser matches upstream's Python output at correlation 0.9999
  (`DenoiserParityTest`).
- The recorder's Vietnamese passage is in `ui/RecordDialog.kt` (`PASSAGE`).

## Windows app

- `setup_windows_gpu.bat` installs into `.venv`: the locked core (`uv sync --frozen`),
  then PyTorch 2.8 (cu126 wheels, which still support GTX 10-series; cu128 for RTX 50xx),
  `transformers==4.57.6`, `safetensors`, `hf_xet`.
- `run_windows.bat` sets `HF_HOME=.hf-home` (models stay on D:, git-ignored), takes the
  first free port from 7860 and opens the browser.
- `src/vieneu/gpu_support.py` picks GPU settings: cards below compute capability 7.0
  (GTX 1060) run float32 (no fast fp16/bf16); under 8 GB of VRAM, batch 8 and 4 streams.
- Speed on the GTX 1060, v3 Turbo: real-time factor 0.44 on the GPU against 1.66 on the
  CPU (`apps/bench_windows.py`).
- `python -m apps.book_batch BOOK_OR_FOLDER [--out DIR] [--voice NAME] [--format mp3|wav|flac] [--paragraph-pause S]`
  is the command-line form of the book tab; output resumes by skipping finished parts.

## Building and testing on this PC (DESKTOP-5QJBV8M)

This machine needs care:

- **8 GB of RAM.** Do not run the Windows app and a Gradle build at the same time. The
  Claude session dropped twice under memory pressure during builds. Gradle is limited
  in `D:\work\audio\tools\gradle-home\gradle.properties` (`-Xmx2g`, Kotlin daemon 1 GB,
  2 workers) and the build script uses `--no-daemon`.
- **C: is nearly full.** Keep caches and temp files on D:.
- **Rust** uses the GNU toolchain (`stable-x86_64-pc-windows-gnu`, no Visual Studio).
  The MinGW folders of Git and Strawberry Perl on `PATH` break Rust build scripts, so
  the build script removes them for its own run.

`D:\work\tools\vieneu-android.ps1` (outside the repo) sets all of that up (JDK 21 in
`D:\work\audio\tools\jdk-21`, Android SDK, model files in `D:\work\audio\vieneu-nano`):

```powershell
# unit tests, including the real-model and phonemiser tests
& D:\work\tools\vieneu-android.ps1 -Tasks ':app:testDebugUnitTest' -Log D:\work\tools\android-test.log -HostG2p
# APK -> android\app\build\outputs\apk\debug\app-debug.apk
& D:\work\tools\vieneu-android.ps1 -Tasks ':app:assembleDebug' -Log D:\work\tools\android-apk.log
```

Without `-HostG2p` the two `SeaG2pTest` tests fail with "no reader_g2p in
java.library.path"; that is the missing PC-side phonemiser, not a bug.
`READER_NANO_DIR` turns on `NanoModelTest`; `READER_DENOISE_PARITY` (a folder with
`denoise-in.f32` and `denoise-python.f32`) turns on `DenoiserParityTest`.

Python tests: `.venv\Scripts\python.exe -m pytest tests/test_book_batch.py`.

Installing on the phone: USB debugging is not enabled, so the APK is copied to the
phone's `Download` folder over the file-transfer (MTP) connection and installed from
the file manager. With USB debugging on (HyperOS: tap "OS version" 7 times, then
Additional settings → Developer options), `adb install -r` works and logs can be read.

## Open items and ideas

- Exercise on the phone: the recorder (permission, level meter, auto-stop at 45 s),
  rename, the pause dialog, prepare-from-chapter. None of these has been tried on a
  device yet.
- `[n]` tags are removed from speech but still shown in the reader's text.
- Release build: R8 shrinking and arm64-only would take the APK from about 120 MB to
  an estimated 40–45 MB (not built yet; most of the size is ONNX Runtime for two ABIs
  and unshrunk dex).
- The Windows pause option is a single paragraph-pause slider plus `[n]` tags; per-mark
  pauses as on the phone are not there yet.
- GitHub Actions results for `b9b4286` were not checked (no `gh` on this PC).
