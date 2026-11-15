# Handoff: VieNeu Reader (Android) and the Windows GPU app

State as of 2026-11-15: `main` at `b9b4286` on `bunyu/VieNeu-TTS` (a fork of
`pnnbao97/VieNeu-TTS`, Apache-2.0), and the work described under "Branch
`claude/reader-fast-start`" below on that branch, pushed but not merged. Start a new
feature from `main` on a branch of its own; push to `main` only once the work is
finished and tested.

## In progress

Nothing is in progress.

## Branch `claude/reader-fast-start` (2026-11-15)

Built and unit-tested on the PC; **nothing on this branch has been run on the phone
yet** except the speed readout (the Xiaomi 14T showed 1.8–2× listening speed in normal
mode and 3–3.5× in fast mode, so v3 Nano keeps ahead of the listener there).

Reader (Android):

- **Plain reader first.** The book screen opens without the player (`Settings.listening`,
  the headphone button in the top bar). With listening off nothing is generated
  (`Narration.reload` starts the render loop only when listening, playing or preparing
  the whole book) and the place follows the scroll (`Narration.readAt`). Text is shown
  in paragraphs (`OpenBook.paragraphs`), one continuous scroll through the book; bars
  hide on a tap; themes, type, margins, brightness and keep-awake are in the "Aa" sheet
  (`AppearanceSheet`); bookmarks live in `books/<id>/bookmarks.json`.
- **Player.** Folds to one row (the arrow beside the status line). A piece the listener
  is waiting for is played part by part as it is generated (`EarlyParts` in `Renderer`,
  files in `cacheDir/early-parts`), its first part cut to the opening sentence
  (`Chunker.calls`). When playback runs dry on a phone slower than listening, it holds
  until `Narration.HOLD_PIECES` are ready (`holdFrom`). Below 1× the player offers fast
  mode. Threads come from the count of fast cores (`BigCores`).
- **Voices.** A play button per preset speaks a sample (`VieNeu.sample`, cached). The
  20 v3 Turbo voices Nano lacks are packed as `assets/voices_extra.json`: each Turbo
  voice spoke a passage on the PC and Nano enrolled that speech
  (`android/app/scripts/make_extra_voices.py`; the clips are in
  `D:\work\audio\extra-voices`, `NN.wav` from Turbo and `NN-nano.wav` from the enrolled
  voice). They are Nano clones of those voices, not the Turbo model. Nobody has listened
  to them yet.
- **Audio made on a PC.** Library menu → "Add audio made on a PC" (pick the audio files
  and the `.sync.json` beside each, or a zip of them; Google Drive shows in the picker)
  or "from a link" (`PreparedAudio.download`; Drive and Dropbox share links are
  rewritten, tested only against a local server). Such a book is `prepared.json` plus
  `prepared/NNNN.ext`; its pieces are the paragraphs of the sync files, each a `Clip`
  played as a clipped ExoPlayer item, so highlight, seek, speed and the sleep timer work
  as for generated audio. Parts can be added later and in any order.

- **Release build, named "Doc truyen".** `:app:assembleRelease` writes
  `app/build/outputs/apk/release/Doc truyen-release.apk`: not debuggable, arm64 only,
  not shrunk, and signed with this PC's debug key so it installs over the debug builds
  and keeps their data. The app's label is "Doc truyen" in both languages.
- **First sound.** A piece someone is waiting for is made in growing parts (opening
  sentence, about 70 characters, then full size; `Chunker.calls`), the first with half
  the steps; that audio is played and thrown away, and the piece is generated properly
  the next time. Pressing play where nothing is ready restarts the piece that way.
  Settings → "Keep the voice ready while reading" generates the two passages at the
  reading place two seconds after the scroll stops (`readyWhileReading`).
- **Shared Drive folder.** A link to a Drive folder is searched for audio, sync and zip
  files (`PreparedAudio.fetchDriveFolder`, one level of sub-folders); parts the book
  already has are not downloaded again. The list comes from Drive's
  `embeddedfolderview` page, which Google does not promise. Checked on 2026-11-15
  with the same patterns against a public test folder (the 100-file example folder of
  the gdown project: all 100 entries listed, sub-folders told apart, a small file
  downloaded through the rewritten link). Not tried: the app itself on the phone, a
  folder of more than 100 files, and a large audio file (Drive may answer with a
  "cannot scan for viruses" page for big files; the link carries `confirm=t` for that).
- **Interface.** Settings → language (phone / Tiếng Việt / English, applied by
  `ReaderSettings.localized` in `MainActivity.attachBaseContext`, then the activity is
  rebuilt; the notification text still follows the phone) and light / dark
  (`AppTheme`, `LocalAppDark`). The library has three buttons: add book, audio files,
  audio link.

Windows app:

- `run_windows.bat` sets `VIENEU_AUTOLOAD=1`: the default model loads when the page
  opens (`autoload_model`). The model files were always kept in `.hf-home`; what each
  start needed was the click on the load button.
- "🔊 Nghe thử giọng này" beside the voice lists (`voice_sample`).
- Book tab: chapters per file (`chapters_per_file`, files named `001-005 - …`), a pause
  per punctuation mark (`pause_rules`, `split_marks`; 0 keeps the natural flow), format
  `opus`, and a `.sync.json` beside every audio file with the time of each paragraph.
  Each paragraph is now spoken by itself, so the paragraph pause really falls between
  paragraphs. Audio made earlier has no sync file; delete a file to have it made again.

Not done:

- **v3 Turbo on the phone** (asked for: a choice between Nano and Turbo). Not started.
  It needs a Kotlin port of `src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py`
  (prefill / decode-step / acoustic graphs with a KV cache, the heads in an `.npz`,
  sampling over 16 codebooks per frame, a byte-level BPE tokenizer, the MOSS codec at
  48 kHz), a download manifest (the fp32 backbone alone is 415 MB; with heads and codec
  about 520 MB) and real-model tests. Measured on this PC (i7-6700, 4 threads, ONNX on
  the CPU, same passage): Nano 3.64× listening speed, Turbo fp32 0.88×, so Turbo is 4.1
  times slower. The 14T makes Nano speech at 1.8–2×, which puts Turbo there at about
  0.45×: a night of eight hours would prepare under four hours of audio, with the CPU
  flat out. Audio made on the PC (GTX 1060: 2.3× listening speed) and imported gives
  the same voice today, so the port was left until that has been tried.
- Smaller or quantised Nano graphs and the XNNPACK backend were not tried.

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
  rename, the pause dialog, prepare-from-chapter, and everything on the branch above
  (first of all: play-as-generated at a seek, the end of a book, and a book imported
  with PC audio, where the joins between clipped stretches need a listen).
- `[n]` tags are removed from speech but still shown in the reader's text.
- Release build: R8 shrinking and arm64-only would take the APK from about 120 MB to
  an estimated 40–45 MB (not built yet; most of the size is ONNX Runtime for two ABIs
  and unshrunk dex).
- The Windows pause option is a single paragraph-pause slider plus `[n]` tags; per-mark
  pauses as on the phone are not there yet.
- GitHub Actions results for `b9b4286` were not checked (no `gh` on this PC).
