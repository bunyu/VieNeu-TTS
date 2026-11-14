package io.github.bunyu.reader.playback

import android.content.Context
import android.net.ConnectivityManager
import io.github.bunyu.reader.media.AacWriter
import io.github.bunyu.reader.media.LocalAudio
import io.github.bunyu.reader.vieneu.NanoPreset
import java.security.MessageDigest
import io.github.bunyu.reader.vieneu.NanoAssets
import io.github.bunyu.reader.vieneu.NanoEnroller
import io.github.bunyu.reader.vieneu.NanoEngine
import io.github.bunyu.reader.vieneu.NanoOptions
import io.github.bunyu.reader.vieneu.NanoPresets
import io.github.bunyu.reader.vieneu.NanoRelease
import io.github.bunyu.reader.vieneu.NanoVoice
import io.github.bunyu.reader.vieneu.ReleaseFile
import io.github.bunyu.reader.vieneu.SeaG2p
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * The app's voice engine: VieNeu-TTS v3 Nano with its sea-g2p phonemiser, both running on the phone.
 * The model (about 280 MB) and the dictionary are downloaded once from Settings; after that reading is offline.
 */
class VieNeu(private val context: Context) {
    val release: NanoRelease = context.assets.open(RELEASE).bufferedReader().use { NanoRelease.decode(it.readText()) }
    val assets = NanoAssets(File(context.filesDir, "vieneu"), release, unmetered = {
        context.getSystemService(ConnectivityManager::class.java)?.let { !it.isActiveNetworkMetered } ?: false
    })
    private val mutex = Mutex()
    private var loaded: Loaded? = null
    private val meter = SpeedMeter()
    @Volatile private var meterFast = false
    private val mutableSpeed = MutableStateFlow<Float?>(null)
    /** Seconds of speech generated per second of work on this phone, or null before the first passage. */
    val speed: StateFlow<Float?> = mutableSpeed.asStateFlow()

    private class Loaded(val engine: NanoEngine, val g2p: SeaG2p, val presets: NanoPresets)

    val ready get() = assets.installed(ReleaseFile.SPEECH)

    private suspend fun load(): Loaded = mutex.withLock {
        loaded ?: withContext(Dispatchers.IO) {
            check(ready) { "model not installed" }
            // Big cores only: more threads land on the efficiency cores and slow every step down.
            val threads = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)
            // A model or native library that cannot load fails every passage the same way: report it once instead of skipping the book.
            try { Loaded(NanoEngine(assets.directory, threads), SeaG2p(assets.file(DICTIONARY)), NanoPresets.read(assets.file(PRESETS)) + extraVoices()) }
            catch (e: Throwable) { throw VoiceUnavailable("runtime_failed") }
        }.also { loaded = it }
    }

    suspend fun presets(): NanoPresets = load().presets

    /**
     * More voices packed with the app: the voices of VieNeu-TTS v3 Turbo, enrolled for this model on a PC
     * (`app/scripts/make_extra_voices.py`). A missing or damaged file only means fewer voices.
     */
    private fun extraVoices(): NanoPresets =
        runCatching { context.assets.open(EXTRA_VOICES).bufferedReader().use { NanoPresets.parse(it.readText()) } }.getOrDefault(NanoPresets(emptyList(), null))

    /** A short sentence in [preset]'s voice, generated once and kept in the cache, for choosing a voice by ear. */
    suspend fun sample(preset: NanoPreset): File = withContext(Dispatchers.Default) {
        val name = MessageDigest.getInstance("SHA-256").digest("${release.id}|${preset.name}|$SAMPLE".toByteArray()).take(12).joinToString("") { "%02x".format(it) }
        val file = File(File(context.cacheDir, "voice-samples"), "$name.m4a")
        if (!file.isFile) {
            val samples = AudioPolish.finish(speak(SAMPLE, preset.voice, fast = false, seed = 0) { false }, NanoEngine.SAMPLE_RATE, 0)
            file.parentFile?.mkdirs()
            val pending = File(file.parentFile, "$name.pending")
            AacWriter.write(LocalAudio(samples, NanoEngine.SAMPLE_RATE), pending)
            check(pending.renameTo(file))
        }
        file
    }

    /** Spoken-form text to 24 kHz samples. Text longer than [NanoEngine.MAX_CHARACTERS] must be split first. */
    suspend fun speak(text: String, voice: NanoVoice, fast: Boolean, seed: Long, cancelled: () -> Boolean): FloatArray {
        val model = load()
        return withContext(Dispatchers.Default) {
            try {
                val started = System.nanoTime()
                model.engine.speak(model.g2p.phonemize(text), voice, options(fast), seed, cancelled).also { samples ->
                    // The two modes differ about twofold, so a measurement never mixes them.
                    if (fast != meterFast) { meter.reset(); meterFast = fast }
                    mutableSpeed.value = meter.add(samples.size.toDouble() / NanoEngine.SAMPLE_RATE, (System.nanoTime() - started) / 1e9)
                }
            }
            // Stopping mid-passage is a cancellation, not a failed passage.
            catch (e: InterruptedException) { throw kotlinx.coroutines.CancellationException("cancelled") }
        }
    }

    val canClone get() = assets.installed(ReleaseFile.CLONE)

    /**
     * A recording to a voice. The three enrolment graphs are loaded for this call only, since a voice is
     * enrolled once and they would otherwise hold memory the reading model needs.
     */
    suspend fun enroll(recording: LocalAudio): NanoVoice = withContext(Dispatchers.Default) {
        check(ready && canClone) { "cloning model not installed" }
        NanoEnroller(assets.directory).use { it.enroll(recording.samples, recording.sampleRate) }
    }

    /** Frees the model's memory, and its files when [delete] is set. */
    suspend fun unload(delete: Boolean = false) {
        mutex.withLock { loaded?.let { it.engine.close(); it.g2p.close() }; loaded = null }
        if (delete) assets.delete()
    }

    companion object {
        const val RELEASE = "vieneu-release.json"
        const val DICTIONARY = "sea_g2p.bin"
        const val PRESETS = "voices_v3_nano.json"
        const val EXTRA_VOICES = "voices_extra.json"
        /** What every voice says as its sample; already in spoken form, so it needs no text rules. */
        const val SAMPLE = "Xin chào, tôi sẽ đọc sách cho bạn nghe bằng giọng này. Chúc bạn có những phút giây thật thư giãn."
        /** Upstream's two settings: 16 Euler steps, or 8 with sway sampling for about twice the speed. */
        fun options(fast: Boolean) = if (fast) NanoOptions(steps = 8, sway = -1f) else NanoOptions()
    }
}
