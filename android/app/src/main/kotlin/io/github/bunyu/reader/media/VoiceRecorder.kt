package io.github.bunyu.reader.media

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max

/**
 * Records the microphone for voice cloning: mono 16-bit at [RATE], at most [MAX_SECONDS]. The caller must hold
 * the RECORD_AUDIO permission. [level] is the recent peak (0 to 1) for a meter, [seconds] the length so far.
 */
class VoiceRecorder {
    private val levelState = MutableStateFlow(0f)
    private val secondsState = MutableStateFlow(0.0)
    val level: StateFlow<Float> = levelState.asStateFlow()
    val seconds: StateFlow<Double> = secondsState.asStateFlow()
    @Volatile private var running = false
    private var thread: Thread? = null
    private val pcm = ByteArrayOutputStream()

    val recording get() = running

    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        pcm.reset(); secondsState.value = 0.0
        val minimum = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val record = AudioRecord(MediaRecorder.AudioSource.MIC, RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, max(minimum, RATE / 5 * 2))
        check(record.state == AudioRecord.STATE_INITIALIZED) { "microphone unavailable" }
        running = true
        thread = Thread {
            val buffer = ShortArray(RATE / 20)
            val bytes = ByteBuffer.allocate(buffer.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            try {
                record.startRecording()
                var total = 0
                while (running && total < MAX_SECONDS * RATE) {
                    val n = record.read(buffer, 0, buffer.size)
                    if (n <= 0) continue
                    bytes.clear()
                    var peak = 0
                    for (i in 0 until n) { bytes.putShort(buffer[i]); peak = max(peak, abs(buffer[i].toInt())) }
                    synchronized(pcm) { pcm.write(bytes.array(), 0, n * 2) }
                    total += n
                    levelState.value = peak / 32768f
                    secondsState.value = total.toDouble() / RATE
                }
            } finally {
                runCatching { record.stop() }; record.release()
                running = false; levelState.value = 0f
            }
        }.apply { name = "voice-recorder"; start() }
    }

    fun stop() { running = false; thread?.join(2000); thread = null }

    /** Writes what was recorded as a WAV file. */
    fun save(target: File): File {
        val data = synchronized(pcm) { pcm.toByteArray() }
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            .put("RIFF".toByteArray()).putInt(36 + data.size).put("WAVEfmt ".toByteArray()).putInt(16).putShort(1).putShort(1)
            .putInt(RATE).putInt(RATE * 2).putShort(2).putShort(16).put("data".toByteArray()).putInt(data.size)
        target.parentFile?.mkdirs()
        target.outputStream().use { it.write(header.array()); it.write(data) }
        return target
    }

    companion object {
        const val RATE = 48_000
        const val MAX_SECONDS = 45
    }
}
