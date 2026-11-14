package io.github.bunyu.reader.media

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import io.github.bunyu.reader.vieneu.SpeechFeatures
import java.io.File
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt

/** Mono samples in -1..1. */
class LocalAudio(val samples: FloatArray, val sampleRate: Int) {
    init { require(sampleRate > 0) }
    val seconds get() = samples.size.toDouble() / sampleRate
}

/** Decodes any audio file the phone can play to mono samples at [sampleRate], keeping at most [maxSeconds]. */
fun interface AudioDecoder {
    fun decode(source: File, sampleRate: Int, maxSeconds: Int): LocalAudio
}

/** The platform decoders (MediaExtractor and MediaCodec): MP3, AAC/M4A, WAV, FLAC, Ogg/Opus and whatever else the phone supports. */
class PlatformAudioDecoder : AudioDecoder {
    override fun decode(source: File, sampleRate: Int, maxSeconds: Int): LocalAudio {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(source.absolutePath)
            val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
                ?: throw IllegalArgumentException("no audio track")
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            val mono = FloatList()
            try {
                codec.configure(format, null, null, 0)
                codec.start()
                val info = MediaCodec.BufferInfo()
                var inputDone = false
                var outputDone = false
                var limit = Long.MAX_VALUE
                while (!outputDone) {
                    if (!inputDone) {
                        val index = codec.dequeueInputBuffer(10_000)
                        if (index >= 0) {
                            val size = extractor.readSampleData(codec.getInputBuffer(index)!!, 0)
                            if (size < 0) { codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone = true }
                            else { codec.queueInputBuffer(index, 0, size, extractor.sampleTime, 0); extractor.advance() }
                        }
                    }
                    val out = codec.dequeueOutputBuffer(info, 10_000)
                    when {
                        out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            rate = codec.outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            channels = codec.outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        }
                        out >= 0 -> {
                            limit = maxSeconds.toLong() * rate
                            val pcm = codec.getOutputBuffer(out)!!.order(ByteOrder.nativeOrder())
                            pcm.position(info.offset); pcm.limit(info.offset + info.size)
                            val floats = codec.outputFormat.let { it.containsKey(MediaFormat.KEY_PCM_ENCODING) && it.getInteger(MediaFormat.KEY_PCM_ENCODING) == android.media.AudioFormat.ENCODING_PCM_FLOAT }
                            if (floats) { val b = pcm.asFloatBuffer(); while (b.remaining() >= channels) { var s = 0f; repeat(channels) { s += b.get() }; mono.add(s / channels) } }
                            else { val b = pcm.asShortBuffer(); while (b.remaining() >= channels) { var s = 0f; repeat(channels) { s += b.get() / 32768f }; mono.add(s / channels) } }
                            codec.releaseOutputBuffer(out, false)
                            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0 || mono.size >= limit) outputDone = true
                        }
                    }
                }
            } finally { runCatching { codec.stop() }; codec.release() }
            val kept = mono.toArray(min(mono.size.toLong(), maxSeconds.toLong() * rate).toInt())
            return LocalAudio(SpeechFeatures.resample(kept, rate, sampleRate), sampleRate)
        } finally { extractor.release() }
    }

    private class FloatList {
        private var data = FloatArray(1 shl 16)
        var size = 0; private set
        fun add(value: Float) { if (size == data.size) data = data.copyOf(size * 2); data[size++] = value }
        fun toArray(count: Int) = data.copyOf(count)
    }
}

/** Writes mono samples as AAC-LC in an MP4 (.m4a) container with the platform encoder. */
object AacWriter {
    fun write(audio: LocalAudio, target: File, bitRate: Int = 64_000) {
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, audio.sampleRate, 1).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16_384)
        }
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        val muxer = MediaMuxer(target.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var started = false
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            val info = MediaCodec.BufferInfo()
            var track = -1
            var position = 0
            var inputDone = false
            while (true) {
                if (!inputDone) {
                    val index = codec.dequeueInputBuffer(10_000)
                    if (index >= 0) {
                        val buffer = codec.getInputBuffer(index)!!.order(ByteOrder.nativeOrder())
                        buffer.clear()
                        val count = min(buffer.remaining() / 2, audio.samples.size - position)
                        val time = position * 1_000_000L / audio.sampleRate
                        for (i in 0 until count) buffer.putShort((audio.samples[position + i].coerceIn(-1f, 1f) * 32767f).roundToInt().toShort())
                        position += count
                        if (position >= audio.samples.size) { codec.queueInputBuffer(index, 0, count * 2, time, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone = true }
                        else codec.queueInputBuffer(index, 0, count * 2, time, 0)
                    }
                }
                val out = codec.dequeueOutputBuffer(info, 10_000)
                if (out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) { track = muxer.addTrack(codec.outputFormat); muxer.start(); started = true }
                else if (out >= 0) {
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && info.size > 0 && started) {
                        val data = codec.getOutputBuffer(out)!!
                        data.position(info.offset); data.limit(info.offset + info.size)
                        muxer.writeSampleData(track, data, info)
                    }
                    codec.releaseOutputBuffer(out, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
        } finally {
            runCatching { codec.stop() }; codec.release()
            if (started) runCatching { muxer.stop() }
            muxer.release()
        }
        check(started && target.length() > 0) { "AAC encoding produced no audio" }
    }
}
