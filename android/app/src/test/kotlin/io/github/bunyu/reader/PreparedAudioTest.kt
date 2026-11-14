package io.github.bunyu.reader

import io.github.bunyu.reader.data.BookSummary
import io.github.bunyu.reader.data.LinkFailed
import io.github.bunyu.reader.data.PreparedAudio
import io.github.bunyu.reader.data.PreparedBook
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class PreparedAudioTest {
    @get:Rule val temp = TemporaryFolder()
    private val json = Json { ignoreUnknownKeys = true }

    /** A sync file as `apps/book_batch.py` writes it: [chapters] of two paragraphs, one second each. */
    private fun sync(index: Int, audio: String, vararg chapters: String): String {
        var at = 0
        val body = chapters.joinToString(",") { title ->
            val segments = (1..2).joinToString(",") { n -> """[${at++}.0, $at.0, "$title, đoạn $n"]""" }
            """{"title": "$title", "segments": [$segments]}"""
        }
        return """{"format": "vieneu-sync", "version": 1, "book": "Truyện thử", "index": $index, "of": 3, "audio": "$audio", "seconds": $at.0, "chapters": [$body]}"""
    }

    private fun stage(vararg files: Pair<String, String>) = temp.newFolder().also { folder -> files.forEach { (name, text) -> File(folder, name).writeText(text) } }
    private fun summary(root: File, id: String) = json.decodeFromString(BookSummary.serializer(), File(File(root, id), "summary.json").readText())

    @Test fun partsBecomeABookThatGrowsAndKeepsItsPlace() {
        val root = temp.newFolder("books")
        // Part 2 arrives first, with a sync file whose audio is missing beside it.
        val first = PreparedAudio.import(root, stage("002 - Ba.opus" to "audio-2", "002 - Ba.opus.sync.json" to sync(2, "002 - Ba.opus", "Ba", "Bốn"),
            "003 - Năm.opus.sync.json" to sync(3, "003 - Năm.opus", "Năm"), "notes.txt" to "x"), now = 5)
        assertEquals(1, first.parts); assertEquals(1, first.withoutAudio)
        val id = first.books.single()
        val folder = File(root, id)
        var book = json.decodeFromString(PreparedBook.serializer(), File(folder, "prepared.json").readText())
        var pieces = book.pieces(File(folder, "prepared"))
        assertEquals(listOf("Ba, đoạn 1", "Ba, đoạn 2", "Bốn, đoạn 1", "Bốn, đoạn 2"), pieces.map { it.text })
        assertEquals(listOf(0, 0, 1, 1), pieces.map { it.chapter })
        assertEquals("audio-2", pieces[2].clip!!.file.readText())
        assertEquals(2000L to 3000L, pieces[2].clip!!.startMs to pieces[2].clip!!.endMs)
        assertEquals(listOf("Ba", "Bốn"), summary(root, id).chapterTitles)

        // The reader is at "Bốn, đoạn 1" when part 1 is added in front, in a zip.
        File(folder, "summary.json").writeText(json.encodeToString(BookSummary.serializer(), summary(root, id).copy(chapter = 1, piece = 2)))
        val staging = temp.newFolder()
        ZipOutputStream(File(staging, "phan-1.zip").outputStream()).use { zip ->
            zip.putNextEntry(ZipEntry("out/001 - Một.opus")); zip.write("audio-1".toByteArray()); zip.closeEntry()
            zip.putNextEntry(ZipEntry("../001 - Một.opus.sync.json")); zip.write(sync(1, "001 - Một.opus", "Một", "Hai").toByteArray()); zip.closeEntry()
        }
        val second = PreparedAudio.import(root, staging, now = 9)
        assertEquals(listOf(id), second.books)
        book = json.decodeFromString(PreparedBook.serializer(), File(folder, "prepared.json").readText())
        pieces = book.pieces(File(folder, "prepared"))
        assertEquals(listOf(1, 2), book.parts.map { it.index })
        assertEquals(8, pieces.size)
        assertEquals("audio-1", pieces[0].clip!!.file.readText())
        val place = summary(root, id)
        assertEquals("Bốn, đoạn 1", pieces[place.piece].text)
        assertEquals(3, place.chapter); assertEquals(8, place.totalPieces); assertEquals(5L, place.addedAt)
        assertEquals(listOf("Một", "Hai", "Ba", "Bốn"), book.stored(id).chapters.map { it.title })
        assertTrue("nothing escapes the staging folder", !File(staging.parentFile, "001 - Một.opus.sync.json").exists())
    }

    @Test fun shareLinksBecomeDownloadLinks() {
        val drive = "https://drive.usercontent.google.com/download?id=1AbC_dEfGhIjKlMnOp-12345&export=download&confirm=t"
        assertEquals(drive, PreparedAudio.directLink(" https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOp-12345/view?usp=sharing "))
        assertEquals(drive, PreparedAudio.directLink("https://drive.google.com/open?id=1AbC_dEfGhIjKlMnOp-12345"))
        assertEquals(drive, PreparedAudio.directLink("https://drive.google.com/uc?export=download&id=1AbC_dEfGhIjKlMnOp-12345"))
        assertEquals("https://www.dropbox.com/s/abc/sach.zip?dl=1", PreparedAudio.directLink("https://www.dropbox.com/s/abc/sach.zip?dl=0"))
        assertEquals("https://example.com/sach.zip", PreparedAudio.directLink("https://example.com/sach.zip"))
        assertEquals("Chương 1.opus", PreparedAudio.fileName("attachment; filename=\"x.opus\"; filename*=UTF-8''Ch%C6%B0%C6%A1ng%201.opus"))
        assertEquals("a b.zip", PreparedAudio.fileName("attachment; filename=\"a b.zip\""))
    }

    @Test fun downloadFollowsRedirectsAndRefusesAPage() {
        // A few canned answers over a plain socket: a redirect, a file, a web page; anything else is not found.
        val server = ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"))
        val serving = Thread {
            while (true) {
                val socket = try { server.accept() } catch (e: Exception) { break }
                socket.use {
                    val path = it.getInputStream().bufferedReader().readLine().orEmpty().split(' ').getOrElse(1) { "" }
                    val (head, body) = when (path) {
                        "/share" -> "HTTP/1.1 302 Found\r\nLocation: /file\r\n" to ByteArray(0)
                        "/file" -> "HTTP/1.1 200 OK\r\nContent-Type: application/zip\r\nContent-Disposition: attachment; filename=\"../sach.zip\"\r\n" to byteArrayOf(1, 2, 3)
                        "/page" -> "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n" to "ok".toByteArray()
                        else -> "HTTP/1.1 404 Not Found\r\n" to ByteArray(0)
                    }
                    it.getOutputStream().apply { write((head + "Content-Length: ${body.size}\r\nConnection: close\r\n\r\n").toByteArray()); write(body); flush() }
                }
            }
        }.apply { isDaemon = true; start() }
        try {
            val base = "http://127.0.0.1:${server.localPort}"
            val into = temp.newFolder()
            val file = PreparedAudio.download("$base/share", into)
            assertEquals(File(into, "sach.zip"), file)
            assertEquals(3L, file.length())
            assertThrows(LinkFailed::class.java) { PreparedAudio.download("$base/page", into) }
            assertThrows(LinkFailed::class.java) { PreparedAudio.download("$base/missing", into) }
        } finally { server.close(); serving.join(2000) }
    }
}
