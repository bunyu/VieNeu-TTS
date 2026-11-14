package io.github.bunyu.reader.vieneu

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Removes room noise from a recording before it is enrolled, as upstream does (`vieneu/_v3_turbo_engine/
 * onnx_denoiser.py`, the resemble-enhance denoiser): a 1680-point STFT at 44.1 kHz, the learned mask in
 * `denoiser.onnx`, and an overlap-add inverse STFT. Plain JVM code, so it is tested off the phone.
 */
class NanoDenoiser(file: File, threads: Int = 4) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setInterOpNumThreads(1)
        setIntraOpNumThreads(threads.coerceIn(1, 8))
    }
    private val session = environment.createSession(file.absolutePath, options)

    /** Any rate in, denoised mono at [RATE] out (at most [MAX_SECONDS]). */
    @Synchronized
    fun denoise(samples: FloatArray, sampleRate: Int): FloatArray {
        val input = SpeechFeatures.resample(samples, sampleRate, RATE).let { it.copyOf(min(it.size, MAX_SECONDS * RATE)) }
        val length = input.size
        val peak = max(input.maxOfOrNull { abs(it) } ?: 0f, 1e-7f)
        // Upstream pads 441 samples (resemble-enhance's inference chunk), then normalises to the peak again.
        val padded = DoubleArray(length + 441).also { for (i in 0 until length) it[i] = input[i] / peak.toDouble() }
        val second = max(padded.maxOf { abs(it) }, 0.0) + 1e-7
        for (i in padded.indices) padded[i] /= second
        val spectrum = Stft.forward(padded)
        val bins = Stft.BINS; val frames = spectrum.frames
        val magnitude = FloatArray(bins * frames); val cosine = FloatArray(bins * frames); val sine = FloatArray(bins * frames)
        for (f in 0 until bins) for (t in 0 until frames) {
            val re = spectrum.real[f * frames + t]; val im = spectrum.imaginary[f * frames + t]
            val angle = atan2(im, re)
            magnitude[f * frames + t] = hypot(re, im).toFloat(); cosine[f * frames + t] = cos(angle).toFloat(); sine[f * frames + t] = sin(angle).toFloat()
        }
        val shape = longArrayOf(1, bins.toLong(), frames.toLong())
        val separated = OnnxTensor.createTensor(environment, FloatBuffer.wrap(magnitude), shape).use { m ->
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(cosine), shape).use { c ->
                OnnxTensor.createTensor(environment, FloatBuffer.wrap(sine), shape).use { s ->
                    session.run(mapOf("mag" to m, "cos" to c, "sin" to s)).use { out ->
                        Array(3) { (out[it] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { a -> b.get(a) } } }
                    }
                }
            }
        }
        val (sepMagnitude, sepCosine, sepSine) = separated
        val real = DoubleArray(bins * frames) { sepMagnitude[it].toDouble() * sepCosine[it] }
        val imaginary = DoubleArray(bins * frames) { sepMagnitude[it].toDouble() * sepSine[it] }
        val output = Stft.inverse(Stft.Spectrum(real, imaginary, frames), padded.size)
        return FloatArray(length) { (output[it] * peak).toFloat() }
    }

    @Synchronized override fun close() { session.close(); options.close() }

    companion object {
        const val FILE = "denoiser.onnx"
        const val RATE = 44_100
        const val MAX_SECONDS = 30

        /**
         * A rough signal-to-noise ratio in dB: the level of loud (90th percentile) against quiet (10th percentile)
         * 20 ms frames. Measured on VieNeu's reference clips, denoising helps below about 30 dB and slightly hurts
         * a clean studio recording, so enrolment only denoises below [NOISY_BELOW_DB].
         */
        fun signalToNoise(samples: FloatArray, sampleRate: Int): Double {
            val hop = max(1, sampleRate / 50)
            val levels = (0 until samples.size - hop step hop).map { start ->
                var sum = 0.0
                for (i in start until start + hop) sum += samples[i].toDouble() * samples[i]
                sqrt(sum / hop + 1e-12)
            }.sorted()
            if (levels.size < 10) return 100.0
            fun percentile(p: Double) = levels[((levels.size - 1) * p).toInt()]
            return 20 * log10(percentile(0.9) / percentile(0.1))
        }

        const val NOISY_BELOW_DB = 30.0
    }
}

/** torch.stft(n_fft = 1680, hop = 420, centre, reflect padding, periodic Hann) and its overlap-add inverse, as upstream. */
internal object Stft {
    const val SIZE = 1680
    const val HOP = 420
    const val BINS = SIZE / 2 + 1
    private val window = DoubleArray(SIZE) { 0.5 - 0.5 * cos(2 * PI * it / SIZE) }

    class Spectrum(val real: DoubleArray, val imaginary: DoubleArray, val frames: Int)

    /** Bins × frames, row-major by bin; the last frame is dropped as upstream does. */
    fun forward(x: DoubleArray): Spectrum {
        val pad = SIZE / 2
        val padded = DoubleArray(x.size + 2 * pad) { i ->
            val j = i - pad
            when { j < 0 -> x[-j]; j >= x.size -> x[2 * (x.size - 1) - j]; else -> x[j] }
        }
        val count = 1 + (padded.size - SIZE) / HOP
        val frames = count - 1
        val real = DoubleArray(BINS * frames); val imaginary = DoubleArray(BINS * frames)
        val re = DoubleArray(SIZE); val im = DoubleArray(SIZE)
        for (t in 0 until frames) {
            for (n in 0 until SIZE) { re[n] = padded[t * HOP + n] * window[n]; im[n] = 0.0 }
            Fft.transform(re, im, inverse = false)
            for (f in 0 until BINS) { real[f * frames + t] = re[f]; imaginary[f * frames + t] = im[f] }
        }
        return Spectrum(real, imaginary, frames)
    }

    fun inverse(spectrum: Spectrum, length: Int): DoubleArray {
        val frames = spectrum.frames + 1  // the last frame is repeated, as upstream
        val total = SIZE + HOP * (frames - 1)
        val out = DoubleArray(total); val weight = DoubleArray(total)
        val re = DoubleArray(SIZE); val im = DoubleArray(SIZE)
        for (t in 0 until frames) {
            val source = min(t, spectrum.frames - 1)
            for (f in 0 until BINS) { re[f] = spectrum.real[f * spectrum.frames + source]; im[f] = spectrum.imaginary[f * spectrum.frames + source] }
            // irfft ignores the imaginary part of the DC and Nyquist bins.
            im[0] = 0.0; im[SIZE / 2] = 0.0
            for (f in BINS until SIZE) { re[f] = re[SIZE - f]; im[f] = -im[SIZE - f] }
            Fft.transform(re, im, inverse = true)
            for (n in 0 until SIZE) { out[t * HOP + n] += re[n] / SIZE * window[n]; weight[t * HOP + n] += window[n] * window[n] }
        }
        val pad = SIZE / 2
        return DoubleArray(length) { i -> val j = i + pad; if (j < total - pad) out[j] / max(weight[j], 1e-11) else 0.0 }
    }
}

/** Mixed-radix FFT for any length (recursive Cooley–Tukey on the smallest factor; 1680 = 2⁴·3·5·7). */
internal object Fft {
    private class Twiddles(val cos: DoubleArray, val sin: DoubleArray)
    private val tables = java.util.concurrent.ConcurrentHashMap<Int, Twiddles>()

    fun transform(real: DoubleArray, imaginary: DoubleArray, inverse: Boolean) {
        val n = real.size
        val table = tables.getOrPut(n) { Twiddles(DoubleArray(n) { cos(2 * PI * it / n) }, DoubleArray(n) { -sin(2 * PI * it / n) }) }
        val (r, i) = recurse(real, imaginary, n, 0, 1, table, if (inverse) -1.0 else 1.0)
        r.copyInto(real); i.copyInto(imaginary)
    }

    /** [sign] flips the table's e^(-iθ) into e^(+iθ) for the inverse transform. */
    private fun recurse(re: DoubleArray, im: DoubleArray, n: Int, offset: Int, stride: Int, table: Twiddles, sign: Double): Pair<DoubleArray, DoubleArray> {
        if (n == 1) return doubleArrayOf(re[offset]) to doubleArrayOf(im[offset])
        val p = smallestFactor(n)
        val m = n / p
        val parts = Array(p) { recurse(re, im, m, offset + it * stride, stride * p, table, sign) }
        val full = table.cos.size
        val scale = full / n
        val outR = DoubleArray(n); val outI = DoubleArray(n)
        for (k in 0 until n) {
            var sr = 0.0; var si = 0.0
            for (q in 0 until p) {
                val (pr, pi) = parts[q]
                val index = ((q.toLong() * k % n) * scale).toInt()
                val c = table.cos[index]; val s = sign * table.sin[index]
                val xr = pr[k % m]; val xi = pi[k % m]
                sr += xr * c - xi * s; si += xr * s + xi * c
            }
            outR[k] = sr; outI[k] = si
        }
        return outR to outI
    }

    private fun smallestFactor(n: Int): Int { var f = 2; while (f * f <= n) { if (n % f == 0) return f; f++ }; return n }
}
