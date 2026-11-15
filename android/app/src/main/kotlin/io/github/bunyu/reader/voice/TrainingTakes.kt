package io.github.bunyu.reader.voice

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.OutputStream
import java.security.MessageDigest
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** One recording kept for fine-tuning: [text] is exactly what was read. A "reading" holds several sentences in a row. */
@Serializable data class TrainingTake(val file: String, val text: String, val kind: String = SENTENCE, val seconds: Double) {
    companion object { const val SENTENCE = "sentence"; const val READING = "reading" }
}
@Serializable data class TrainingSet(val name: String, val takes: List<TrainingTake> = emptyList()) {
    val seconds get() = takes.sumOf { it.seconds }
}

/**
 * Recordings of one person kept, with their text, for fine-tuning a voice on the PC (the Windows app's tab
 * "🎓 Fine-tune giọng"). A set per voice name; [export] writes it as a `.dtdata` file:
 *
 *     dataset.json  {"format": "doc-truyen-dataset", "version": 1, "voice": name,
 *                    "takes": [{"file": "0001.wav", "text": "...", "kind": "sentence" | "reading"}]}
 *     0001.wav …    16-bit mono WAV, as recorded
 */
class TrainingTakes(private val root: File) {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val mutex = Mutex()
    private val state = MutableStateFlow<List<TrainingSet>>(emptyList())
    val sets: StateFlow<List<TrainingSet>> = state.asStateFlow()

    /** A folder per name, named by a hash so any name is safe on disk. */
    private fun dir(name: String): File {
        val key = MessageDigest.getInstance("SHA-256").digest(name.trim().lowercase().toByteArray()).take(8).joinToString("") { "%02x".format(it) }
        return File(root, key)
    }
    private fun read(folder: File): TrainingSet? = runCatching { json.decodeFromString(TrainingSet.serializer(), File(folder, FILE).readText()) }.getOrNull()

    suspend fun refresh() = withContext(Dispatchers.IO) {
        mutex.withLock { state.value = root.listFiles().orEmpty().mapNotNull(::read).filter { it.takes.isNotEmpty() }.sortedBy { it.name.lowercase() } }
    }

    fun set(name: String): TrainingSet? = state.value.firstOrNull { it.name.trim().equals(name.trim(), ignoreCase = true) }

    /** Keeps a copy of [wav] (a WAV file) read as [text]. Reading the same sentence again replaces the earlier take. */
    suspend fun add(name: String, text: String, wav: File, kind: String = TrainingTake.SENTENCE) = withContext(Dispatchers.IO) {
        val voice = name.trim().take(60)
        require(voice.isNotEmpty() && text.isNotBlank())
        mutex.withLock {
            val folder = dir(voice).apply { mkdirs() }
            val current = read(folder) ?: TrainingSet(voice)
            val words = text.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
            val old = current.takes.firstOrNull { it.text == words }
            val file = old?.file ?: "%04d.wav".format((current.takes.maxOfOrNull { it.file.substringBefore('.').toIntOrNull() ?: 0 } ?: 0) + 1)
            wav.copyTo(File(folder, file), overwrite = true)
            val take = TrainingTake(file, words, kind, wavSeconds(File(folder, file)))
            val next = current.copy(name = current.name, takes = current.takes.filter { it.file != file } + take)
            File(folder, FILE).writeText(json.encodeToString(TrainingSet.serializer(), next))
        }
        refresh()
    }

    /** Writes the set [name] as a `.dtdata` file to [out]. */
    suspend fun export(name: String, out: OutputStream) = withContext(Dispatchers.IO) {
        val folder = dir(name)
        val set = read(folder) ?: error("no recordings")
        ZipOutputStream(out.buffered()).use { zip ->
            zip.putNextEntry(ZipEntry("dataset.json"))
            val takes = set.takes.joinToString(",") { t -> """{"file": ${quote(t.file)}, "text": ${quote(t.text)}, "kind": ${quote(t.kind)}}""" }
            zip.write("""{"format": "doc-truyen-dataset", "version": 1, "voice": ${quote(set.name)}, "takes": [$takes]}""".toByteArray())
            zip.closeEntry()
            for (take in set.takes) {
                val file = File(folder, take.file).takeIf { it.isFile } ?: continue
                zip.putNextEntry(ZipEntry(take.file)); file.inputStream().use { it.copyTo(zip) }; zip.closeEntry()
            }
        }
    }

    suspend fun delete(name: String) = withContext(Dispatchers.IO) { mutex.withLock { dir(name).deleteRecursively() }; refresh() }

    private fun quote(text: String) = kotlinx.serialization.json.JsonPrimitive(text).toString()

    companion object {
        const val SUFFIX = ".dtdata"
        private const val FILE = "takes.json"

        /** Seconds of a 16-bit mono WAV, from its header. */
        fun wavSeconds(file: File): Double = runCatching {
            val head = file.inputStream().use { it.readNBytes(44) }
            val rate = java.nio.ByteBuffer.wrap(head, 24, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).int
            val channels = java.nio.ByteBuffer.wrap(head, 22, 2).order(java.nio.ByteOrder.LITTLE_ENDIAN).short
            (file.length() - 44).toDouble() / (rate * 2 * channels)
        }.getOrDefault(0.0)
    }
}
