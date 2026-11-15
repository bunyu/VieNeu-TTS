# Handoff: the Windows app and the "Doc truyen" phone app

For the next session. The user-facing guide is the top-level `README.md`; this note holds
what a developer needs: the state, how things work, what was checked and what was not.

## State (2026-11-15)

- Repository `bunyu/VieNeu-TTS`, a fork of `pnnbao97/VieNeu-TTS` (Apache-2.0).
- **All the work below is on the branch `claude/reader-fast-start`, pushed, not merged.**
  `main` is still at `b9b4286`. The rule agreed with the user: merge to `main` only after
  the phone build has been tried on the phone.
- The user's phone is a **Xiaomi 14T** (HyperOS, USB debugging off: APKs go through the
  phone's `Download` folder). The PC is an i7-6700 with 8 GB RAM and a GTX 1060 6 GB.
- The user asked that long work continue on Claude's recommendation without waiting
  for answers, and that progress stay resumable through this note: keep
  "In progress" current.

## In progress

Nothing is in progress.

## What to do first next time

The phone has run almost none of this branch. Ask the user to install
`android/app/build/outputs/apk/release/Doc truyen-release.apk` and try, in this order:

1. Press play at a place with nothing prepared; seek while playing; play to the end of a book.
2. Import a short book made on the PC in `opus` with its `.sync.json` files (from files,
   then from a Drive folder link) and listen to the joins between paragraphs.
3. Import a `.dtvoice` made by the PC (Voices → From a file); listen to a few of the 20
   extra built-in voices.
4. Language switch, light/dark, bookmarks, "Keep the voice ready while reading".

Measured on the phone so far: Nano makes speech at 1.8–2× listening speed in normal mode
and 3–3.5× in fast mode. Before the first-sound changes the user heard a 1–2 s wait when
pressing play where nothing was ready; after them, not yet measured.

## Windows app (`apps/`, `run_windows.bat`)

- **Install / run.** `setup_windows_gpu.bat` → `.venv` with PyTorch 2.8 (cu126 wheels for
  GTX 9xx/10xx, cu128 for RTX 50xx) and `transformers==4.57.6`. `run_windows.bat` sets
  `HF_HOME=.hf-home` (models downloaded once, git-ignored), `VIENEU_AUTOLOAD=1` (the
  default model loads when the page opens, `autoload_model`), takes the first free port
  from 7860 and opens the browser. GPU settings come from `src/vieneu/gpu_support.py`:
  below compute capability 7.0 float32; under 8 GB batch 8 and 4 streams.
- **Speed (GTX 1060, v3 Turbo):** 2.3× listening speed on the GPU, 0.6× on the CPU.
- **Voice sample** buttons (`voice_sample`), cached per model and voice for the session.
- **Book → audio** (`apps/book_batch.py`, also a command line). Each paragraph is spoken
  by itself (`render_piece`, batched on the GPU through `infer_batch`), followed by the
  paragraph pause. With per-mark pause rules a paragraph is first cut after those marks
  (`split_marks`; never after abbreviations, decimals or initials) and the model's own
  trailing silence is trimmed (`trim_tail`) before the exact pause. `chapters_per_file`
  groups parts (`group_parts`, names `001-005 - <first title>`), with 1.4 s between
  chapters. Formats mp3 / opus / wav / flac. Beside every file goes
  `<file>.sync.json` (format below). Finished files are skipped on a rerun; files made
  before sync files existed are kept and reported, not remade.
- **Clone comparison** (`apps/voice_lab.py`). Up to six 8-second stretches that are nearly
  all speech (≥ 85 % of the best, overlapping by half at most, moved to a phrase start),
  each cloned and reading the same passage, ranked by cosine similarity of Turbo speaker
  vectors against up to 60 s of the whole recording. Checked with the real model: clones
  of a built-in voice 0.92–0.94, a different speaker 0.38.
- **Fine-tuning** (`apps/finetune_studio.py`, `finetune_windows.bat`). The tab records one
  sentence at a time into `finetune/dataset/<voice>/` (`raw_audio/NNNN.wav`,
  `metadata.csv`, `takes.json`; a retake replaces the take; silence trimmed; 3–15 s
  enforced). The batch file (app closed) installs peft / accelerate / pyarrow with
  `uv pip` once, then runs `prepare_dataset.py` → `train_lora.py` (on cards older than
  RTX 30 or under 7 GB: `--no-bf16 --batch-size 2 --grad-accum 8 --grad-checkpoint`) →
  `merge_lora.py` **as its own process** (merging inside the training process hit a
  `MemoryError` on 8 GB) → `make_voice.py --default` with the longest take of at most 8 s.
  The app lists every `finetune/output/<voice>/merged` as "VieNeu-TTS-v3-Turbo 🎓 <voice>"
  (GPU only). Fixed upstream code: `train_lora.py` turned gradient checkpointing on
  before attaching LoRA, and PEFT then crashed in `get_input_embeddings`; it is now
  switched on afterwards, non-reentrant. Checked with a 20-clip trial next to the running
  app: about 10 s per optimizer step of 16 clips, `acc_cb0` 0.41, merge 2 min 40 s, the
  merged model loads in 34 s and speaks. **No real voice has been trained yet.**
- **Phone voices** (`apps/phone_voice.py`, accordion "📱 Dùng giọng trên app điện thoại").
  The chosen voice of the loaded Turbo model reads a ~10 s passage; Nano of the phone's
  release (files from `android/app/src/main/assets/vieneu-release.json`, downloaded with
  sha-256 checks into `.hf-home/phone-nano`, or `VIENEU_NANO_DIR`) enrols it without the
  denoiser; Nano reads a check line. Output `~/Music/Doc truyen - giong/<name>.dtvoice`.
  Checked with the real models ("Thiện Minh"): 25 s, a 242 KB file; the phone version's
  speech scores 0.85 against the original voice (Turbo speaker vectors), a different Nano
  voice 0.41. Not yet imported on the phone.
- **Phone recordings for fine-tuning.** The recording box also takes an uploaded file. A long
  recording of several listed sentences, read in order with about a second of pause after
  each, is cut at the pauses (`split_reading`: 10 ms frames, quiet = 30 dB under the 95th
  percentile or twice the 10th, gaps ≥ 0.6 s) and paired in order (`import_reading`).
  Nothing is saved if any piece's syllables per second fall outside 0.7–1.45× the median
  (a forgotten pause or a skipped sentence); pieces outside 3–15 s are left out and listed.
  `load_audio` reads phone formats through PyAV (`av`, now installed by
  `setup_windows_gpu.bat`; it carries its own ffmpeg). Checked on real speech: six
  sentences read by Turbo with 1 s pauses, saved as `.m4a`, all paired correctly, the
  2.5 s one left out. Not yet tried with a real phone recording of a person.
- `apps/bench_windows.py` measures speed; `tests/clone_similarity.py` measures Nano clones.

## Doc truyen (`android/`, module `:app`)

`android/README.md` describes the code in more depth. Map of
`app/src/main/kotlin/io/github/bunyu/reader/`:

- `vieneu/`: `NanoEngine` (port of upstream `v3nano.py`), `NanoEnroller`, `NanoDenoiser`,
  `SpeechFeatures`, `SeaG2p` (JNI to the Rust phonemiser in `app/native/g2p`),
  `NanoAssets` / `NanoRelease` (pinned downloads), `NanoPresets` (the model's 11 voices
  plus `assets/voices_extra.json`, the 20 Turbo voices enrolled for Nano by
  `app/scripts/make_extra_voices.py`; their clips are in `D:\work\audio\extra-voices`).
- `playback/`: `Narration` (queue, render-ahead loop, hold after running dry, sleep
  timer, music), `Renderer` (piece → cached `.m4a`; `EarlyParts` for a piece someone is
  waiting for), `SpeedMeter`, `BigCores` (threads = cores within 75 % of the fastest's
  top frequency, 2–6), `VieNeu` (model, samples), `ReaderService`, `AudioPolish`.
- `data/`: `Library` (books, positions, bookmarks, imports), `PreparedAudio` (PC audio:
  sync files, zips, links, Drive folders), `ReaderSettings`.
- `text/`: `Chunker` (pieces; `calls` = model calls, growing when someone waits),
  `VietnameseText` / `VietnameseNumbers`, `PauseRules` / `PauseText`.
- `voice/`: `ReaderVoices` (`print.f32` = 192 speaker floats then 50 × 256 style floats,
  little-endian, plus a preview clip), `VoiceBundle` (VoiceStudio), `PcVoiceFile`
  (`.dtvoice`).
- `ui/`: Compose screens; `Theme` (`AppTheme`, `LocalAppDark`, reading themes).
- Strings: `res/values/strings.xml` (English) and `res/values-vi/strings.xml`; every new
  string needs both. The app label is "Doc truyen" (without accents, as the user wrote it).

### Behaviour worth knowing before changing it

- **Plain reader first.** `Settings.listening` (headphone button) shows the player. With it
  off nothing is generated unless "prepare whole book" or `readyWhileReading` is on (two
  passages at the reading place, 2 s after the scroll stops); the place follows the
  scroll (`Narration.readAt`).
- **Audio cache.** `filesDir/books/<id>/audio/<key>.m4a`. The key hashes the model
  release, fast mode, `VietnameseText.VERSION`, dialect, pronunciations, the voice and
  its `enrolledAt`, the changed pause rules, and the piece text and pause. Anything that
  changes the sound must join the key; pause rules join only when changed, so default
  audio stays valid. Bump `VietnameseText.VERSION` when normalisation changes speech.
- **First sound.** A piece someone waits for is made in calls of growing size (opening
  sentence ≥ 30 characters, ~70, then full), the first with 8 steps; those parts are
  played from `cacheDir/early-parts` and **not cached**: the piece is generated properly
  next time. `play()` restarts the piece this way when nothing is queued.
- **Hold.** When playback runs dry and the measured speed is below 1×, it waits for
  `HOLD_PIECES` (3) before resuming, and offers fast mode.
- **Pauses.** A mark at its default keeps the model's flow; a changed mark splits there
  and is followed by exactly that silence (`PauseText.segments`, `Renderer.render`,
  `endPause`). `[n]` tags (≤ 10 s each, they add up) are not spoken but are still shown in
  the text. `AudioPolish` keeps ~60 ms of breath around each part.
- **Cloning on the phone.** `NanoEnroller` denoises only below 30 dB SNR and takes style
  from the most voiced 5 s. Similarity of cloned speech to the speaker
  (`tests/clone_similarity.py`, 5 clips in `examples/audio_ref`):

  | Reference | No denoise | Denoise | Denoise only when noisy (shipped) |
  |---|---|---|---|
  | clean | 0.779 | 0.760 | 0.773 |
  | room noise, SNR 25 dB | 0.711 | 0.721 | 0.721 |
  | room noise, SNR 15 dB | 0.659 | 0.705 | 0.705 |

- **PC audio books** are `books/<id>/prepared.json` + `prepared/NNNN.<ext>`; their pieces are
  the paragraphs of the sync files, each a `Clip` played as a clipped ExoPlayer item (MP3
  with index seeking). Parts arrive in any order; the reader's place is kept as "this
  paragraph of that part". Such books need no voice model.
- **Drive folders** are listed from `https://drive.google.com/embeddedfolderview?id=<id>`
  (blocks `class="flip-entry"`, title in `flip-entry-title`, link `/file/d/<id>` or
  `/folders/<id>`), one level of sub-folders, sync files first so known parts are skipped.
  Google does not promise that page. Checked against a public 100-file folder (all listed,
  one file downloaded through `drive.usercontent.google.com/download?…&confirm=t`). Not
  tried: in the app, more than 100 files, large files.
- **Interface language** is applied in `MainActivity.attachBaseContext`
  (`ReaderSettings.localized`) and the activity is recreated; the notification channel
  name still follows the phone.
- **Release build.** `:app:assembleRelease` → `Doc truyen-release.apk` (68 MB): x86_64
  libraries excluded for release only (`androidComponents` packaging excludes), not
  shrunk (R8 untried: ONNX Runtime and the phonemiser are reached through JNI), signed
  with this PC's debug key so it installs over the debug builds and keeps their data.

## File formats shared by the PC and the phone

- **`<audio>.sync.json`** (`apps/book_batch.py` → `PreparedAudio.parse`):
  `{"format": "vieneu-sync", "version": 1, "book", "index", "of", "audio", "seconds",
  "chapters": [{"title", "segments": [[start_s, end_s, "paragraph text"], …]}]}`. A
  segment spans the paragraph and its pause; the book is matched by its title.
- **`.dtvoice`** (`apps/phone_voice.py` → `PcVoiceFile`): a zip of `voice.json`
  `{"format": "doc-truyen-voice", "version": 1, "name", "description", "release",
  "speaker": [192], "style": [50 × [256]]}` and `sample.wav` (Nano reading a check line).
  The phone refuses a file whose `release` is not its own model release.

## Building and testing on this PC

- **8 GB of RAM.** Don't run a Gradle build and the Windows app's model at the same time if
  avoidable (the session dropped twice under memory pressure); Gradle is limited in
  `D:\work\audio\tools\gradle-home\gradle.properties` and runs `--no-daemon`. Training
  needs the app closed (6 GB card).
- **C: is nearly full**: caches and temp files stay on D:.
- **Rust** uses the GNU toolchain; the build script removes Git's and Strawberry Perl's
  MinGW folders from `PATH` for its run.
- `D:\work\tools\vieneu-android.ps1` (outside the repo) sets up JDK 21, the SDK and the
  model folder `D:\work\audio\vieneu-nano` (the phone's Nano release, sha-checked):

  ```powershell
  & D:\work\tools\vieneu-android.ps1 -Tasks ':app:testDebugUnitTest',':app:assembleRelease' -Log D:\work\tools\android-test.log -HostG2p
  ```

  A test-plus-release run takes 6–13 minutes. Without `-HostG2p` the two `SeaG2pTest`
  tests fail (no PC phonemiser). `READER_NANO_DIR` turns on `NanoModelTest`, which can
  fail with an ONNX allocation error when memory is short (it did once beside the app).
- Python tests: `.venv\Scripts\python.exe -m pytest tests/test_book_batch.py tests/test_voice_tools.py tests/test_phone_voice.py tests/test_finetune_lora.py`.
- Gradio 5.49.1. Gradle on this PC writes files with LF; git normalises line endings, so a
  `.bat` file needs CRLF in the working copy (write it with `newline='\r\n'`).

## Decisions and measurements

- **v3 Turbo on the phone was not ported.** Measured on this PC's CPU with 4 threads: Nano
  3.64× listening speed, Turbo fp32 0.88× (4.1× slower). On the 14T that puts Turbo near
  0.45×. A port needs the Turbo ONNX engine in Kotlin (prefill / decode-step / acoustic
  graphs with a KV cache, 16 codebooks per frame, byte-level BPE, the MOSS codec at
  48 kHz) and about 520 MB of downloads. Turbo audio made on the PC and imported, and
  `.dtvoice` voices, cover the need for now.
- **No true streaming on the phone**: Nano generates a whole stretch at once (flow
  matching), so the first-sound work makes the first stretch short instead.

## Open items

- Everything under "What to do first next time".
- Fine-tuning from free recordings (words not known beforehand) would need speech
  recognition, for example a Vietnamese Whisper model (about 1.5 GB) plus a screen to
  correct each transcript. Offered to the user on 2026-11-15; not asked for yet.
- Smaller or quantised Nano graphs and the XNNPACK backend: not tried.
- R8 shrinking of the release: not tried.
- `[n]` tags are still shown in the reader's text.
- The notification text does not follow the in-app language.
- GitHub Actions results were not checked (no `gh` on this PC).
