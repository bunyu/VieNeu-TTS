package io.github.bunyu.reader

import io.github.bunyu.reader.media.AudioDecoder
import io.github.bunyu.reader.voice.PcVoiceException
import io.github.bunyu.reader.voice.PcVoiceFile
import io.github.bunyu.reader.voice.ReaderVoices
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class PcVoiceTest {
    @get:Rule val temp = TemporaryFolder()

    /** A file as apps/phone_voice.py writes it. */
    private fun voiceFile(release: String = "r1", speakers: Int = 192, extra: String = ""): File {
        val speaker = (0 until speakers).joinToString(",") { "${it / 100.0}" }
        val row = (0 until 256).joinToString(",") { "0.5" }
        val style = (0 until 50).joinToString(",") { "[$row]" }
        val json = """{"format": "doc-truyen-voice", "version": 1, "name": "Giọng bố", "description": "trầm", "release": "$release", "speaker": [$speaker], "style": [$style]$extra}"""
        return File(temp.newFolder(), "Giong bo.dtvoice").also { file ->
            ZipOutputStream(file.outputStream()).use { zip ->
                zip.putNextEntry(ZipEntry("voice.json")); zip.write(json.toByteArray()); zip.closeEntry()
                zip.putNextEntry(ZipEntry("sample.wav")); zip.write(byteArrayOf(1, 2, 3)); zip.closeEntry()
            }
        }
    }

    @Test fun aVoiceFromThePcIsKeptWithoutEnrolling() = runBlocking {
        val pc = PcVoiceFile.read(voiceFile(), temp.newFolder("work"), "r1")
        assertEquals("Giọng bố", pc.name); assertEquals("trầm", pc.description)
        assertEquals(0.01f, pc.voice.speaker[1]); assertEquals(50, pc.voice.styleTokens); assertEquals(0.5f, pc.voice.style[12_799])
        assertEquals(3L, pc.sample!!.length())
        val noDecoding = AudioDecoder { _, _, _ -> error("must not decode") }
        val voices = ReaderVoices(temp.newFolder("voices"), noDecoding)
        val stored = voices.addEnrolled(pc.name, pc.voice, pc.sample)
        assertEquals(listOf("Giọng bố"), voices.voices.value.map { it.name })
        val back = voices.print(stored)!!
        assertArrayEquals(pc.voice.speaker, back.speaker, 0f); assertArrayEquals(pc.voice.style, back.style, 0f)
        assertEquals("preview.wav", voices.preview(stored)!!.name)
    }

    @Test fun otherFilesAndOtherModelsAreRefused() {
        val other = assertThrows(PcVoiceException::class.java) { PcVoiceFile.read(voiceFile(release = "old"), temp.newFolder(), "r1") }
        assertEquals(PcVoiceException.Reason.OTHER_MODEL, other.reason)
        val short = assertThrows(PcVoiceException::class.java) { PcVoiceFile.read(voiceFile(speakers = 100), temp.newFolder(), "r1") }
        assertEquals(PcVoiceException.Reason.NOT_A_VOICE, short.reason)
        val text = temp.newFile("x.dtvoice").apply { writeText("not a zip") }
        assertThrows(PcVoiceException::class.java) { PcVoiceFile.read(text, temp.newFolder(), "r1") }
    }
}
