package io.github.bunyu.reader

import io.github.bunyu.reader.vieneu.Fft
import io.github.bunyu.reader.vieneu.NanoDenoiser
import io.github.bunyu.reader.vieneu.NanoEnroller
import io.github.bunyu.reader.vieneu.Stft
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class NanoDenoiserTest {
    @Test fun fftMatchesTheDefinitionForTheDenoiserLength() {
        val n = 1680
        val random = Random(3)
        val re = DoubleArray(n) { random.nextGaussian() }; val im = DoubleArray(n) { random.nextGaussian() }
        val outR = re.copyOf(); val outI = im.copyOf()
        Fft.transform(outR, outI, inverse = false)
        for (k in listOf(0, 1, 7, 420, 839, 840, 1679)) {
            var sr = 0.0; var si = 0.0
            for (t in 0 until n) { val a = -2 * PI * k * t / n; sr += re[t] * cos(a) - im[t] * sin(a); si += re[t] * sin(a) + im[t] * cos(a) }
            assertEquals(sr, outR[k], 1e-6); assertEquals(si, outI[k], 1e-6)
        }
        Fft.transform(outR, outI, inverse = true)
        assertArrayEquals(re, DoubleArray(n) { outR[it] / n }, 1e-9)
    }

    @Test fun stftRoundTripRestoresTheSignal() {
        val random = Random(5)
        val x = DoubleArray(44_100) { sin(2 * PI * 440 * it / 44_100.0) * 0.5 + random.nextGaussian() * 0.01 }
        val back = Stft.inverse(Stft.forward(x), x.size)
        // The dropped last frame leaves the very end approximate, as upstream.
        for (i in 0 until x.size - Stft.SIZE) assertEquals("sample $i", x[i], back[i], 1e-6)
    }

    @Test fun signalToNoiseSeparatesCleanFromNoisy() {
        val rate = 16_000
        val random = Random(1)
        // Speech-like bursts with pauses between them.
        val clean = FloatArray(rate * 4) { i -> if ((i / (rate / 4)) % 2 == 0) (sin(2 * PI * 200 * i / rate) * 0.3).toFloat() else 0f }
        val noisy = FloatArray(clean.size) { clean[it] + (random.nextGaussian() * 0.02).toFloat() }
        assertTrue(NanoDenoiser.signalToNoise(clean, rate) > NanoDenoiser.NOISY_BELOW_DB)
        assertTrue(NanoDenoiser.signalToNoise(noisy, rate) < NanoDenoiser.NOISY_BELOW_DB)
    }

    @Test fun mostVoicedWindowSkipsALongPause() {
        val rate = 1000
        val samples = FloatArray(rate * 9) { i -> if (i < rate * 3) 0.0001f else (sin(i * 0.3) * 0.5).toFloat() }
        val window = NanoEnroller.mostVoiced(samples, rate, 5.0)
        assertEquals(rate * 5, window.size)
        // No sample of the quiet lead-in (exactly 0.0001) is in the chosen window.
        assertTrue(window.none { it == 0.0001f })
        assertTrue(window.count { abs(it) > 0.01f } > rate * 4)
    }
}
