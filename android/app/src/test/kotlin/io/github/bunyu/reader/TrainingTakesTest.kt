package io.github.bunyu.reader

import io.github.bunyu.reader.voice.TrainingTake
import io.github.bunyu.reader.voice.TrainingTakes
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipInputStream

class TrainingTakesTest {
    @get:Rule val temp = TemporaryFolder()

    /** A WAV like VoiceRecorder.save writes: 48 kHz, mono, 16-bit. */
    private fun wav(seconds: Double): File = temp.newFile().also { file ->
        val data = ByteArray((seconds * 48_000).toInt() * 2)
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).put("RIFF".toByteArray()).putInt(36 + data.size).put("WAVEfmt ".toByteArray())
            .putInt(16).putShort(1).putShort(1).putInt(48_000).putInt(96_000).putShort(2).putShort(16).put("data".toByteArray()).putInt(data.size)
        file.writeBytes(header.array() + data)
    }

    @Test fun takesAreKeptPerVoiceAndExportedWithTheirText() = runBlocking {
        val takes = TrainingTakes(temp.newFolder("training"))
        takes.add("Giọng bố", "Câu  thứ nhất.", wav(3.5))
        takes.add("giọng bố ", "Câu thứ hai.", wav(4.0))
        takes.add("Giọng bố", "Câu thứ nhất.", wav(5.0))          // a retake replaces the first
        takes.add("Giọng bố", "Một đoạn dài. Có hai câu.", wav(20.0), TrainingTake.READING)
        takes.add("Mẹ", "Câu của mẹ.", wav(3.2))
        assertEquals(listOf("Giọng bố", "Mẹ"), takes.sets.value.map { it.name })
        val dad = takes.set("GIỌNG BỐ")!!
        assertEquals(listOf("0002.wav" to "Câu thứ hai.", "0001.wav" to "Câu thứ nhất.", "0003.wav" to "Một đoạn dài. Có hai câu."), dad.takes.map { it.file to it.text })
        assertEquals(29.0, dad.seconds, 0.01)

        val out = ByteArrayOutputStream()
        takes.export("Giọng bố", out)
        val members = mutableMapOf<String, ByteArray>()
        ZipInputStream(out.toByteArray().inputStream()).use { zip -> while (true) { val e = zip.nextEntry ?: break; members[e.name] = zip.readBytes() } }
        assertEquals(setOf("dataset.json", "0001.wav", "0002.wav", "0003.wav"), members.keys)
        val data = Json.parseToJsonElement(members.getValue("dataset.json").toString(Charsets.UTF_8)).jsonObject
        assertEquals("doc-truyen-dataset", data.getValue("format").jsonPrimitive.content)
        assertEquals("Giọng bố", data.getValue("voice").jsonPrimitive.content)
        val listed = data.getValue("takes").jsonArray.map { it.jsonObject }
        assertEquals(listOf("sentence", "sentence", "reading"), listed.map { it.getValue("kind").jsonPrimitive.content })
        assertEquals(44 + 5 * 96_000, members.getValue("0001.wav").size)

        takes.delete("Mẹ")
        assertEquals(listOf("Giọng bố"), takes.sets.value.map { it.name })
    }
}
