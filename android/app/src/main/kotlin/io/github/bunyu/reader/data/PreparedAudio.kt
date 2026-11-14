package io.github.bunyu.reader.data

import io.github.bunyu.reader.text.Pause
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLDecoder
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID
import java.util.zip.ZipFile

/** A stretch of an audio file and the paragraph spoken in it, in seconds from the start of the file. */
@Serializable data class PreparedSegment(val start: Double, val end: Double, val text: String)
@Serializable data class PreparedChapter(val title: String, val segments: List<PreparedSegment>)
/** One audio file of a book made on a PC: file [index] of [of]. [file] is its name in the book's `prepared` folder. */
@Serializable data class PreparedPart(val index: Int, val of: Int, val file: String, val chapters: List<PreparedChapter>)

/**
 * A book that came with its audio: made on a PC by `apps/book_batch.py`, which writes a `.sync.json` file
 * beside every audio file. The text shown is the text in those files, so it always matches what is heard.
 * Parts may arrive in any order and at different times; chapters of parts not yet added are simply absent.
 */
@Serializable data class PreparedBook(val title: String, val parts: List<PreparedPart>) {
    fun stored(id: String) = StoredBook(id, title, null, parts.flatMap { it.chapters }.map { c -> StoredChapter(c.title, c.segments.joinToString("\n") { it.text }) })

    /** One piece per paragraph, each tied to its stretch of audio in [folder]. */
    fun pieces(folder: File): List<BookPiece> {
        val result = mutableListOf<BookPiece>()
        var chapter = 0
        for (part in parts) for (c in part.chapters) {
            for (s in c.segments) result += BookPiece(result.size, chapter, s.text, Pause.PARAGRAPH,
                Clip(File(folder, part.file), (s.start * 1000).toLong(), (s.end * 1000).toLong()))
            chapter++
        }
        return result
    }

    internal fun segmentsBefore(index: Int) = parts.filter { it.index < index }.sumOf { p -> p.chapters.sumOf { it.segments.size } }
}

/** The audio of one piece: from [startMs] to [endMs] of [file]. */
class Clip(val file: File, val startMs: Long, val endMs: Long)

class SyncFile(val book: String, val index: Int, val of: Int, val audio: String, val chapters: List<PreparedChapter>)
class LinkFailed(val reason: String) : Exception(reason)

object PreparedAudio {
    const val SUFFIX = ".sync.json"
    const val FILE = "prepared.json"
    const val FOLDER = "prepared"
    private val json = Json { ignoreUnknownKeys = true }

    /** What an import did: the books it added to, how many audio files went in, and sync files whose audio was not among the files. */
    class Result(val books: List<String>, val parts: Int, val withoutAudio: Int)

    fun parse(text: String): SyncFile {
        val root = Json.parseToJsonElement(text).jsonObject
        require(root["format"]?.jsonPrimitive?.content == "vieneu-sync") { "not a sync file" }
        val chapters = root.getValue("chapters").jsonArray.map { c ->
            val chapter = c.jsonObject
            PreparedChapter(chapter.getValue("title").jsonPrimitive.content, chapter.getValue("segments").jsonArray.map { s ->
                val (start, end, words) = s.jsonArray
                PreparedSegment(start.jsonPrimitive.double, end.jsonPrimitive.double, words.jsonPrimitive.content).also { require(it.start >= 0 && it.end >= it.start) }
            })
        }
        return SyncFile(root.getValue("book").jsonPrimitive.content.trim().ifEmpty { "?" }, root.getValue("index").jsonPrimitive.int,
            root.getValue("of").jsonPrimitive.int, File(root.getValue("audio").jsonPrimitive.content).name, chapters)
    }

    /**
     * Takes every audio file in [staging] that has its sync file into the library under [root] (the books
     * folder): a new book, or more parts of one that is already there under the same title. Zip files in
     * [staging] are opened first. Files are moved, so [staging] must be on the same storage.
     */
    fun import(root: File, staging: File, now: Long = System.currentTimeMillis()): Result {
        staging.listFiles().orEmpty().filter { it.isFile && it.name.endsWith(".zip", true) }.forEach { unzip(it, staging); it.delete() }
        val found = staging.listFiles().orEmpty().filter { it.isFile && it.name.endsWith(SUFFIX, true) }.mapNotNull { file ->
            runCatching { parse(file.readText()) }.getOrNull()?.let { it to file }
        }
        var withoutAudio = 0
        val ready = found.mapNotNull { (sync, file) ->
            val audio = listOf(File(staging, sync.audio), File(staging, file.name.dropLast(SUFFIX.length))).firstOrNull { it.isFile }
            if (audio == null) { withoutAudio++; null } else sync to audio
        }
        val books = ready.groupBy { it.first.book }.map { (title, parts) -> merge(root, title, parts, now) }
        return Result(books, ready.size, withoutAudio)
    }

    private fun merge(root: File, title: String, incoming: List<Pair<SyncFile, File>>, now: Long): String {
        val existing = root.listFiles().orEmpty().firstOrNull { folder ->
            runCatching { json.decodeFromString(PreparedBook.serializer(), File(folder, FILE).readText()).title == title }.getOrDefault(false)
        }
        val folder = existing ?: File(root, UUID.randomUUID().toString())
        val audio = File(folder, FOLDER).apply { mkdirs() }
        val old = existing?.let { json.decodeFromString(PreparedBook.serializer(), File(it, FILE).readText()) } ?: PreparedBook(title, emptyList())
        val summary = runCatching { json.decodeFromString(BookSummary.serializer(), File(folder, "summary.json").readText()) }.getOrNull()
        // Where the reader is, as "this paragraph of that part", so parts added in front do not move it.
        val place = summary?.let { s ->
            old.parts.firstNotNullOfOrNull { part ->
                val before = old.segmentsBefore(part.index)
                val size = part.chapters.sumOf { it.segments.size }
                if (s.piece in before until before + size) part.index to s.piece - before else null
            }
        }
        val added = incoming.map { (sync, file) ->
            val name = "%04d.%s".format(sync.index, file.extension.lowercase().filter { it.isLetterOrDigit() }.take(8).ifEmpty { "audio" })
            old.parts.firstOrNull { it.index == sync.index }?.let { File(audio, it.file).delete() }
            Files.move(file.toPath(), File(audio, name).toPath(), StandardCopyOption.REPLACE_EXISTING)
            PreparedPart(sync.index, sync.of, name, sync.chapters)
        }
        val book = PreparedBook(title, (old.parts.filter { p -> added.none { it.index == p.index } } + added).sortedBy { it.index })
        val id = folder.name
        val stored = book.stored(id)
        val pieces = book.pieces(audio)
        val piece = (place?.let { (part, at) -> book.segmentsBefore(part) + at } ?: 0).coerceIn(0, (pieces.size - 1).coerceAtLeast(0))
        write(File(folder, FILE), json.encodeToString(PreparedBook.serializer(), book))
        write(File(folder, "book.json"), json.encodeToString(StoredBook.serializer(), stored))
        write(File(folder, "summary.json"), json.encodeToString(BookSummary.serializer(), BookSummary(id, title, null, summary?.addedAt ?: now,
            stored.chapters.map { it.title }, chapter = pieces.getOrNull(piece)?.chapter ?: 0, piece = piece, totalPieces = pieces.size, openedAt = now)))
        return id
    }

    private fun write(file: File, text: String) {
        val pending = File(file.parentFile, file.name + ".pending")
        pending.writeText(text)
        Files.move(pending.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }

    /** Unpacks [zip] flat into [into]: folders inside the archive are dropped, so a name can never leave [into]. */
    fun unzip(zip: File, into: File) {
        runCatching {
            ZipFile(zip).use { archive ->
                for (entry in archive.entries()) {
                    val name = File(entry.name).name
                    if (entry.isDirectory || name.isEmpty() || name.endsWith(".zip", true)) continue
                    archive.getInputStream(entry).use { input -> File(into, name).outputStream().use { input.copyTo(it, 262_144) } }
                }
            }
        }
    }

    /** Share links of Google Drive and Dropbox, turned into links that answer with the file itself. */
    fun directLink(link: String): String {
        val url = link.trim()
        val drive = Regex("""drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^#]*&)?id=)([\w-]{10,})""").find(url)
        if (drive != null) return "https://drive.usercontent.google.com/download?id=${drive.groupValues[1]}&export=download&confirm=t"
        if ("dropbox.com/" in url) return url.replace(Regex("""([?&])dl=0"""), "$1dl=1").let { if ("dl=1" in it) it else it + (if ('?' in it) "&" else "?") + "dl=1" }
        return url
    }

    /** Downloads the file behind [link] into [into] under the name the server gives it; a web page instead of a file is refused. */
    fun download(link: String, into: File, progress: (Long) -> Unit = {}): File {
        var url = URL(directLink(link))
        repeat(8) {
            val connection = (url.openConnection() as HttpURLConnection).apply { instanceFollowRedirects = false; connectTimeout = 20_000; readTimeout = 60_000 }
            try {
                val code = connection.responseCode
                if (code in 301..308) { url = URL(url, connection.getHeaderField("Location") ?: throw LinkFailed("redirect")); return@repeat }
                if (code != 200) throw LinkFailed("http $code")
                // A private Drive file, or one that needs a click to confirm, answers with a page, not the file.
                if (connection.contentType.orEmpty().startsWith("text/html", true)) throw LinkFailed("page")
                val name = fileName(connection.getHeaderField("Content-Disposition")) ?: File(URLDecoder.decode(url.path, "UTF-8")).name.ifEmpty { "download" }
                val target = File(into.apply { mkdirs() }, File(name).name)
                connection.inputStream.use { input ->
                    target.outputStream().use { out ->
                        val buffer = ByteArray(262_144); var total = 0L
                        while (true) { val n = input.read(buffer); if (n < 0) break; out.write(buffer, 0, n); total += n; progress(total) }
                    }
                }
                return target
            } finally { connection.disconnect() }
        }
        throw LinkFailed("redirect")
    }

    internal fun fileName(disposition: String?): String? {
        if (disposition == null) return null
        Regex("""filename\*\s*=\s*UTF-8''([^;]+)""", RegexOption.IGNORE_CASE).find(disposition)?.let { return runCatching { URLDecoder.decode(it.groupValues[1].trim(), "UTF-8") }.getOrNull() }
        return Regex("""filename\s*=\s*"?([^";]+)"?""", RegexOption.IGNORE_CASE).find(disposition)?.groupValues?.get(1)?.trim()?.takeIf { it.isNotEmpty() }
    }
}
