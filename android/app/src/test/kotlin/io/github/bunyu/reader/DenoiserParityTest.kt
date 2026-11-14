package io.github.bunyu.reader

import io.github.bunyu.reader.vieneu.NanoDenoiser
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Checks the Kotlin denoiser against upstream's Python one: set READER_DENOISE_PARITY to a folder holding
 * `denoise-in.f32` (24 kHz mono, little-endian floats) and `denoise-python.f32` (upstream's 44.1 kHz output).
 */
class DenoiserParityTest {
    private fun floats(file: File) = file.readBytes().let { b -> FloatArray(b.size / 4).also { ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) } }

    @Test fun matchesUpstream() {
        val folder = System.getenv("READER_DENOISE_PARITY")?.let(::File)
        val model = System.getenv("READER_NANO_DIR")?.let { File(it, NanoDenoiser.FILE) }
        assumeTrue(folder != null && model != null && model.isFile && File(folder, "denoise-in.f32").isFile)
        val expected = floats(File(folder, "denoise-python.f32"))
        val actual = NanoDenoiser(model!!).use { it.denoise(floats(File(folder!!, "denoise-in.f32")), 24_000) }
        val n = minOf(expected.size, actual.size)
        var dot = 0.0; var ee = 0.0; var aa = 0.0
        for (i in 0 until n) { dot += expected[i] * actual[i]; ee += expected[i] * expected[i]; aa += actual[i] * actual[i] }
        val correlation = dot / sqrt(ee * aa)
        println("denoiser parity: ${actual.size} vs ${expected.size} samples, correlation %.4f, level ratio %.3f".format(correlation, sqrt(aa / ee)))
        assertTrue("Kotlin denoiser differs from upstream: correlation $correlation", correlation > 0.95)
    }
}
