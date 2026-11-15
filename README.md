# VieNeu-TTS — Windows book narrator and the "Doc truyen" phone app

This is a fork of **[pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)**, the Vietnamese
text-to-speech models by Pham Nguyen Ngoc Bao. For the models themselves, the Python SDK, the API
server, benchmarks and the full model documentation, see that repository.

This fork adds two programs built on those models:

| | What it is | Model | Where |
|---|---|---|---|
| **Windows app** | A web app on your PC: read text aloud, turn whole books into audio files, clone and fine-tune voices, make voices for the phone | VieNeu-TTS **v3 Turbo** (48 kHz) on an NVIDIA GPU | `apps/`, `run_windows.bat` |
| **Doc truyen** (Android) | An EPUB reader that reads aloud on the phone, offline | VieNeu-TTS **v3 Nano** (24 kHz) on the phone's CPU | `android/` |

They work together. The PC can make audio for a whole book and the voice files the phone uses; the
phone plays that audio while following the text.

---

## 1. Windows app

### Install (once)

You need Windows 10 or 11, an NVIDIA graphics card (a GTX 1060 6 GB works) with a recent driver,
and [uv](https://docs.astral.sh/uv/getting-started/installation/). No administrator rights are needed.

```bat
git clone https://github.com/bunyu/VieNeu-TTS.git
cd VieNeu-TTS
setup_windows_gpu.bat
```

`setup_windows_gpu.bat` installs everything into `.venv` next to it, including PyTorch for your
card: CUDA 12.6 wheels, which still run GTX 9xx/10xx cards, or CUDA 12.8 for RTX 50xx.

### Run

Double-click **`run_windows.bat`**. It opens the app in your browser at `http://127.0.0.1:7860`, or
the next free port, and loads the default model, v3 Turbo, on its own. Wait for
"✅ Model đã tải thành công" at the top.

- **No second download.** The models are downloaded once, into `.hf-home` next to the app, and
  reused on every start; only loading them into memory takes the 30–40 seconds.
- **Choosing another model.** The list at the top also holds models you fine-tuned (marked 🎓);
  choose one and press "🔄 Nạp model vào bộ nhớ".
- **Memory.** Close the app before training a voice, or before building the phone app on a PC with
  8 GB of memory.

### What it can do

**Read text aloud** (first tab). Paste text, choose a voice, press "🎵 Bắt đầu". Press
"🔊 Nghe thử giọng này" to hear any voice say a sample sentence before choosing it. The tags `[cười]`,
`[thở dài]` and `[hắng giọng]` in the text are spoken as a laugh, a sigh or a throat-clear
(still experimental).

**Book → audio** (tab "📚 Sách → audio"). Give an `.epub`, a `.txt`, or a folder of parts.

- **Chapters per file:** "📑 Số chương trong mỗi file audio". With 1, every chapter is a file; with
  5, files are `001-005 - …`, `006-010 - …`.
- **Pauses:** "⏸️ Ngắt nghỉ giữa các đoạn" sets the pause between paragraphs. Under "⏸️ Ngắt nghỉ
  theo từng dấu câu" you can make each punctuation mark stop for an exact time (0 keeps the
  natural flow). A tag such as `[0.8]` or `[2]` in the text stops for that many seconds.
- **Format:** `mp3`, `opus`, `wav` or `flac`. Use **opus** for the phone: small files that seek
  exactly.
- **Text sync file:** beside every audio file goes a `.sync.json` with the time of each paragraph.
  Copy both to the phone and the app follows the audio in the text.
- **Resume:** files already done are skipped, so pressing the button again continues where it
  stopped.

Command-line form:

```bat
.venv\Scripts\python.exe -m apps.book_batch "D:\Books\Story.epub" --format opus --chapters-per-file 5 ^
    --voice "Thiện Minh" --paragraph-pause 0.5 --pause ".=0.6" --pause ",=0.2"
```

**Clone a voice** (tab "🎤 Voice Cloning").

- **Quick clone:** upload 3–8 seconds of one person speaking, then press "💾 Lưu giọng" to add it
  to every voice list. Only the first 8 seconds are used, and they set both the timbre and the
  way of reading.
- **Better clones:** open "🔬 Tìm đoạn mẫu tốt nhất từ một bản thu dài" and upload 30 seconds to a
  few minutes of the voice, read the way you want books read. The app cuts up to six 8-second
  stretches, clones each, has each read the same passage, and ranks them by how much they sound
  like the whole recording. Listen, pick one and save it.

**Use a PC voice on the phone** ("📱 Dùng giọng trên app điện thoại", in the cloning tab). See
[section 3](#3-moving-voices-and-audio-from-the-pc-to-the-phone).

**Other tabs:**

- "🎭 Hội thoại": a conversation with several voices.
- SRT: subtitles to speech, keeping their timing.

### Fine-tune a voice (closest to the built-in voices)

A clone only hears 8 seconds. Fine-tuning teaches the model one voice from 10–30 minutes of
recordings, and gets much closer to the quality of the built-in voices. It runs on the same 6 GB card.

1. **Record, in the app.** Open the tab "🎓 Fine-tune giọng", type a voice name, and press
   "▶ Bắt đầu / thu tiếp". It shows one sentence at a time: record it with the microphone button
   and press "💾 Lưu và sang câu tiếp".
   - There are 40 built-in sentences; for more, put the path of an `.epub` or `.txt` in the box
     beside the name, ideally the kind of book you want read.
   - **Recording on a phone instead** works too, in two ways:
     - **One sentence at a time:** upload the phone's recording of the sentence shown into the
       same recording box.
     - **Many sentences in one recording:** read the listed sentences in order, from the one
       shown, with **a pause of about a second after each**, then upload the file under
       "📲 Tải lên một bản ghi âm dài". The app cuts it at the pauses and files each piece under
       its sentence.

     If a piece does not match its sentence (a forgotten pause, a skipped sentence), nothing is
     saved and the app names the piece; a sentence under 3 seconds is left out and listed.
     Phone formats such as `.m4a` are read.
   - Aim for **100–300 sentences** (10–30 minutes), each 3–15 seconds, in a quiet room with the
     same microphone, read the way you want books read. You can stop and continue another day.
   - The recordings are kept in `finetune\dataset\<voice name>\`.
2. **Close the app**: the card cannot hold the app and the training together.
3. **Train.** In the app's folder, run:

   ```bat
   finetune_windows.bat "Voice name"
   ```

   The first time, it installs the training packages (peft, accelerate, pyarrow). Then it prepares
   the recordings, trains, merges and enrols the voice. On a GTX 1060 training takes about
   10 seconds per 16 sentences per pass (about 7 minutes for 200 sentences), plus about
   3 minutes to merge.
4. **Use it.** Start `run_windows.bat`, choose the model "VieNeu-TTS-v3-Turbo 🎓 Voice name", load
   it, and choose the voice of the same name. This model knows only that voice; load the normal
   model for anything else.

The steps underneath are those of [`finetune/README.md`](finetune/README.md) (Vietnamese), which
also lists the training options.

---

## 2. Doc truyen (Android app)

An EPUB and text reader that can read aloud in Vietnamese, offline, with a voice of your choice.
Android 10 or later on a 64-bit (arm64) phone. Developed on a Xiaomi 14T.

### Install

Copy `Doc truyen-release.apk` to the phone (for example its `Download` folder over USB) and open it
in the file manager; allow installing from that app if asked. A new version installs over the old
one and keeps your books, voices and the downloaded model.

Then, in the app: **Settings → Voice model → Download** (about 350 MB, Wi-Fi by default). After
that it reads offline.

### What it can do

**Reading.**

- **Library:** EPUB 2/3 and plain text (UTF-8, UTF-16, Windows-1258); chapters follow the book's
  reading order.
- **Page:** one continuous scroll, with bars that hide on a tap.
- **"Aa" settings:** light, sepia, dark or black pages; text size, typeface, line spacing, margins,
  justified text, brightness and keep the screen on.
- **Bookmarks:** mark a place and return to it from the chapters list.

**Listening.** The headphone button in the top bar shows the player; it folds to one row with the
arrow.

- **Controls:** speed 0.5–2× without a change of pitch, a sleep timer (minutes or end of chapter),
  quiet background music, and lock-screen and headset controls.
- **Pauses:** a pause per punctuation mark, and `[n]` tags in the text.
- **Generated ahead:** passages are generated ahead of you (40 by default) and kept.
  **Prepare whole book** generates everything, for example overnight while charging.
- **Speed on the phone:** the player shows how fast the phone makes speech (a Xiaomi 14T makes
  about 2× listening speed, 3–3.5× with **Faster generation**). When you start where nothing is
  ready, there is a short wait while the first sentence is generated.
- **Ready at once:** **Settings → Keep the voice ready while reading** prepares the passages where
  you are reading, so listening starts at once.

**Voices.**

- **Built-in:** 31 Vietnamese voices: the model's own 11, plus 20 made from the v3 Turbo voices.
  Each has a play button for a sample.
- **Clone:** record yourself, or use a short recording of one speaker.
- **From the PC:** a voice made on the PC (`.dtvoice`, see section 3) or a VoiceStudio export
  (`.ovsvoice`).

**Vietnamese text.**

- **Spoken forms:** numbers, dates, times, money, units, Roman chapter numbers and common
  abbreviations are read the way they are spoken, with northern or southern number words.
- **Pronunciation list** (Settings): `written = spoken`, for names and foreign words.

**Interface.** Vietnamese or English, light or dark (Settings → Interface).

### Build it yourself

Needs JDK 21, the Android SDK and NDK, and Rust (the phonemiser), with the Android targets and
cargo-ndk:

```sh
rustup target add aarch64-linux-android x86_64-linux-android
cargo install cargo-ndk
cd android
./gradlew :app:assembleRelease     # app/build/outputs/apk/release/Doc truyen-release.apk
./gradlew :app:testDebugUnitTest
```

On Windows without Visual Studio, use the GNU Rust toolchain
(`rustup default stable-x86_64-pc-windows-gnu`). The release build is signed with the debug key of
the PC that builds it, so it installs over earlier builds from that PC; it is not meant for a store.
Details of the code: [`android/README.md`](android/README.md).

---

## 3. Moving voices and audio from the PC to the phone

**Turbo quality on the phone: whole books made on the PC.** The PC's v3 Turbo sounds better than
the phone's Nano and generates faster than you listen.

1. **Make the audio:** in "📚 Sách → audio" choose the format **opus**.
2. **Copy it:** put the audio files *and* their `.sync.json` files on the phone, in a Google Drive
   folder shared with "Anyone with the link", or in a zip.
3. **Add it:** in the phone's library, press **Audio files** (pick them, including from Drive in
   the picker) or **Audio link** (paste the link of the Drive folder, or one link per file).

The app shows the book's text and highlights the paragraph being heard. The book can arrive in
parts: add the next files, or use the same folder link again, and only the new parts are
downloaded.

**A PC voice for the phone's own reading.**

1. **Make the voice file:** in the PC app, open "📱 Dùng giọng trên app điện thoại" in the cloning
   tab, choose any voice (a clone, a built-in voice, or the voice of a fine-tuned model), give it a
   name and press "📱 Tạo file giọng cho điện thoại".
2. **Listen and copy:** the app plays the original voice and the phone version side by side, and
   saves a `.dtvoice` file in `Music\Doc truyen - giong`.
3. **Add it on the phone:** **Voices → From a file**.

How it works: the phone reads with Nano, so the voice reads a passage on the PC and Nano learns
it from that speech. The phone version is close to the original, not identical. The first time,
the PC downloads the phone's Nano model (about 380 MB, kept in `.hf-home\phone-nano`).

---

## Repository map

| Path | What |
|---|---|
| `run_windows.bat`, `setup_windows_gpu.bat`, `finetune_windows.bat` | Start, install, fine-tune on Windows |
| `apps/gradio_main.py` | The web app |
| `apps/book_batch.py` | Book → audio files with sync files (also a command line) |
| `apps/voice_lab.py` | Compare clones from one long recording |
| `apps/finetune_studio.py` | Record a fine-tuning dataset; the training runner |
| `apps/phone_voice.py` | Make `.dtvoice` files for the phone |
| `android/` | The Doc truyen app (Kotlin, Compose, ONNX Runtime, Rust phonemiser) |
| `finetune/` | LoRA fine-tuning scripts (from upstream, with a fix for small cards) |
| `src/vieneu/` | The upstream SDK the Windows app uses |
| `docs/HANDOFF.md` | Working notes: state, what was checked, open items |

## Licence and credits

Apache-2.0, as upstream. VieNeu-TTS (v3 Turbo, v3 Nano) and the
[sea-g2p](https://github.com/pnnbao97/sea-g2p) phonemiser are by Pham Nguyen Ngoc Bao
([pnnbao97](https://github.com/pnnbao97)); v3 Turbo uses the
[MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) codec.
The phone app downloads the models from their own releases. Please cite and credit the
[original repository](https://github.com/pnnbao97/VieNeu-TTS) when you use the models.
