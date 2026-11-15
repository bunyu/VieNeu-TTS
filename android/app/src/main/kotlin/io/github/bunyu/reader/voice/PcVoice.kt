package io.github.bunyu.reader.voice

import io.github.bunyu.reader.text.readBounded
import io.github.bunyu.reader.vieneu.NanoVoice
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.util.zip.ZipFile

class PcVoiceException(val reason: Reason) : Exception(reason.name) {
    enum class Reason { NOT_A_VOICE, OTHER_MODEL }
}

/** A voice made by the Windows app for this phone: already enrolled for v3 Nano, with [sample] spoken in it. */
class PcVoice(val name: String, val description: String, val release: String, val voice: NanoVoice, val sample: File?)

/**
 * Reads `.dtvoice` files, which the Windows app writes (`apps/phone_voice.py`): a zip holding `voice.json`
 * (speaker vector and style tokens for v3 Nano) and `sample.wav`. Only those two fixed members are read.
 */
object PcVoiceFile {
    const val SUFFIX = ".dtvoice"
    private const val SPEAKER = 192
    private const val STYLE_TOKENS = 50
    private const val STYLE_WIDTH = 256
    private const val MAX_JSON = 2L * 1024 * 1024
    private const val MAX_SAMPLE = 16L * 1024 * 1024

    /** Opens [source]; the sample, if any, is written into [workspace]. [release] is the phone model's release id. */
    fun read(source: File, workspace: File, release: String): PcVoice {
        val zip = try { ZipFile(source) } catch (e: Exception) { throw PcVoiceException(PcVoiceException.Reason.NOT_A_VOICE) }
        zip.use {
            fun bytes(name: String, limit: Long): ByteArray? {
                val entry = zip.getEntry(name)?.takeIf { !it.isDirectory } ?: return null
                if (entry.size > limit) throw PcVoiceException(PcVoiceException.Reason.NOT_A_VOICE)
                return zip.getInputStream(entry).use { it.readBounded(limit) } ?: throw PcVoiceException(PcVoiceException.Reason.NOT_A_VOICE)
            }
            val text = bytes("voice.json", MAX_JSON) ?: throw PcVoiceException(PcVoiceException.Reason.NOT_A_VOICE)
            val voice = try {
                val root = Json.parseToJsonElement(text.toString(Charsets.UTF_8)).jsonObject
                require(root.getValue("format").jsonPrimitive.content == "doc-truyen-voice" && root.getValue("version").jsonPrimitive.int == 1)
                val speaker = root.getValue("speaker").jsonArray.map { it.jsonPrimitive.float }.toFloatArray()
                val rows = root.getValue("style").jsonArray
                require(speaker.size == SPEAKER && rows.size == STYLE_TOKENS && speaker.all { it.isFinite() })
                val style = FloatArray(STYLE_TOKENS * STYLE_WIDTH)
                rows.forEachIndexed { r, row ->
                    val values = row.jsonArray
                    require(values.size == STYLE_WIDTH)
                    values.forEachIndexed { c, x -> style[r * STYLE_WIDTH + c] = x.jsonPrimitive.float.also { require(it.isFinite()) } }
                }
                fun field(key: String) = (root[key] as? JsonPrimitive)?.contentOrNull?.trim().orEmpty()
                Triple(field("name").ifEmpty { source.nameWithoutExtension }.take(200), field("description").take(200), field("release")) to NanoVoice(speaker, style, STYLE_TOKENS)
            } catch (e: Exception) { throw PcVoiceException(PcVoiceException.Reason.NOT_A_VOICE) }
            val (fields, print) = voice
            // Made with another release of the model, the numbers would mean something else to this one.
            if (fields.third != release) throw PcVoiceException(PcVoiceException.Reason.OTHER_MODEL)
            val sample = bytes("sample.wav", MAX_SAMPLE)?.let { data -> File(workspace.apply { mkdirs() }, "sample.wav").apply { writeBytes(data) } }
            return PcVoice(fields.first, fields.second, fields.third, print, sample)
        }
    }
}
