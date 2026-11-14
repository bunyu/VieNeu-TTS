package io.github.bunyu.reader.playback

import io.github.bunyu.reader.data.BookPiece
import io.github.bunyu.reader.media.AacWriter
import io.github.bunyu.reader.media.LocalAudio
import io.github.bunyu.reader.data.Settings
import io.github.bunyu.reader.data.VoiceChoice
import io.github.bunyu.reader.text.Chunker
import io.github.bunyu.reader.text.Pause
import io.github.bunyu.reader.text.PauseRules
import io.github.bunyu.reader.text.PauseText
import io.github.bunyu.reader.text.VietnameseText
import io.github.bunyu.reader.vieneu.NanoEngine
import io.github.bunyu.reader.vieneu.NanoVoice
import io.github.bunyu.reader.voice.ReaderVoices
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID
import kotlin.coroutines.coroutineContext

/** A voice ready to speak: everything a render needs, resolved once per settings change. */
class ResolvedVoice(val choice: VoiceChoice, val voice: NanoVoice, val rules: VietnameseText, val fast: Boolean, private val identity: String,
    val pauses: PauseRules = PauseRules()) {
    private fun digest(piece: BookPiece): ByteArray {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update("$identity|${piece.pause.millis}|".toByteArray())
        digest.update(piece.text.toByteArray())
        return digest.digest()
    }
    fun key(piece: BookPiece): String = digest(piece).take(16).joinToString("") { "%02x".format(it) }
    /** The starting noise for a piece: fixed, so re-reading a passage sounds the same every time. */
    fun seed(piece: BookPiece, part: Int): Long = ByteBuffer.wrap(digest(piece)).getLong(16) + part
}

class VoiceUnavailable(val reason: String) : Exception(reason)

/** Turns one book piece into a short, normalized, compressed audio file in the book's audio folder. */
class Renderer(private val vieneu: VieNeu, private val voices: ReaderVoices) {
    suspend fun resolve(settings: Settings): ResolvedVoice {
        if (!vieneu.ready) throw VoiceUnavailable("engine_missing")
        val rules = settings.textRules()
        // Pause rules join the key only when changed, so audio prepared with the defaults stays valid.
        val pauses = settings.pauses.key.let { if (it.isEmpty()) "" else "|pauses:$it" }
        val shared = "${vieneu.release.id}|${settings.fast}|${VietnameseText.VERSION}|${settings.dialect}|${settings.pronunciations.hashCode()}$pauses"
        return when (val choice = settings.voice) {
            is VoiceChoice.Cloned -> {
                val voice = voices.voices.value.firstOrNull { it.id == choice.id } ?: throw VoiceUnavailable("voice_missing")
                val print = voices.print(voice) ?: throw VoiceUnavailable("voice_missing")
                ResolvedVoice(choice, print, rules, settings.fast, "clone|${voice.id}|${voice.enrolledAt}|$shared", settings.pauses)
            }
            is VoiceChoice.Preset, null -> {
                val presets = vieneu.presets()
                val preset = (choice as? VoiceChoice.Preset)?.let { presets[it.name] } ?: presets.default ?: throw VoiceUnavailable("voice_missing")
                ResolvedVoice(VoiceChoice.Preset(preset.name), preset.voice, rules, settings.fast, "preset|${preset.name}|$shared", settings.pauses)
            }
        }
    }

    fun file(folder: File, voice: ResolvedVoice, piece: BookPiece) = File(folder, voice.key(piece) + ".m4a")

    suspend fun render(folder: File, voice: ResolvedVoice, piece: BookPiece): File = withContext(Dispatchers.Default) {
        val target = file(folder, voice, piece)
        if (target.isFile) return@withContext target
        folder.mkdirs()
        val rate = NanoEngine.SAMPLE_RATE
        val context = coroutineContext
        val polished = ArrayList<FloatArray>()
        // Exact pauses split the piece first: after marks whose pause the user set, and at [n] tags.
        val segments = PauseText.segments(piece.text, voice.pauses)
        var call = 0
        segments.forEachIndexed { s, segment ->
            val last = s == segments.lastIndex
            val after = if (last) endPause(piece.pause, segment.pauseMillis, voice.pauses) else segment.pauseMillis ?: Pause.SENTENCE.millis
            val spoken = voice.rules.normalize(segment.text)
            // The model reads at most about fifteen seconds in one go, so a long stretch is spoken in parts.
            val parts = if (spoken.none { it.isLetterOrDigit() }) emptyList() else Chunker.split(spoken, maxChars = NanoEngine.MAX_CHARACTERS, minChars = 40)
            if (parts.isEmpty() && segment.pauseMillis != null) polished += FloatArray((rate.toLong() * after / 1000).toInt())
            parts.forEachIndexed { index, part ->
                context.ensureActive()
                val samples = vieneu.speak(part.text, voice.voice, voice.fast, voice.seed(piece, call++)) { !context.isActive }
                polished += AudioPolish.finish(samples, rate, if (index == parts.lastIndex) after else part.pause.millis)
            }
        }
        val joined = FloatArray(polished.sumOf { it.size }).also { out -> var at = 0; for (p in polished) { p.copyInto(out, at); at += p.size } }
        val pending = File(folder, UUID.randomUUID().toString() + ".m4a")
        try {
            AacWriter.write(LocalAudio(if (joined.isEmpty()) FloatArray(rate / 10) else joined, rate), pending)
            Files.move(pending.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            target
        } finally { pending.delete() }
    }
}

/**
 * The pause after a piece's last words: a changed new-line pause applies at a paragraph end (and lengthens a
 * chapter end), otherwise an exact pause from a changed mark or tag, otherwise the natural one.
 */
internal fun endPause(piece: Pause, exact: Int?, rules: PauseRules): Int {
    val newLine = rules.millis(PauseRules.NEW_LINE)
    return when (piece) {
        Pause.PARAGRAPH -> newLine ?: exact ?: piece.millis
        Pause.CHAPTER -> maxOf(piece.millis, newLine ?: 0, exact ?: 0)
        Pause.SENTENCE -> exact ?: piece.millis
    }
}
