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

    @Test fun aSharedDriveFolderIsSearchedAndKnownPartsAreSkipped() {
        assertEquals("1AbC_dEfGhIjKlMnOp-12345", PreparedAudio.driveFolder("https://drive.google.com/drive/folders/1AbC_dEfGhIjKlMnOp-12345?usp=sharing"))
        assertEquals("1AbC_dEfGhIjKlMnOp-12345", PreparedAudio.driveFolder("https://drive.google.com/drive/u/0/folders/1AbC_dEfGhIjKlMnOp-12345"))
        assertEquals(null, PreparedAudio.driveFolder("https://drive.google.com/file/d/1AbC_dEfGhIjKlMnOp-12345/view"))

        // The shape of Drive's embedded folder list: one block per entry, a link and a title in each.
        fun entry(id: String, name: String, folder: Boolean = false) = """<div class="flip-entry" id="entry-$id" tabindex="0" role="link"><div class="flip-entry-info">""" +
            """<a href="https://drive.google.com/${if (folder) "drive/folders/$id" else "file/d/$id/view?usp=drive_web"}" target="_blank"><div class="flip-entry-visual"><img src="x"/></div>""" +
            """<div class="flip-entry-title">$name</div></a></div></div>"""
        val listed = PreparedAudio.parseDriveFolder("<html><body>" + entry("file-id-0000001", "001 - M&#39;ột &amp; Hai.opus") + entry("folder-id-000002", "phần sau", folder = true) + "</body></html>")
        assertEquals(listOf("file-id-0000001" to "001 - M'ột & Hai.opus", "folder-id-000002" to "phần sau"), listed.map { it.id to it.name })
        assertEquals(listOf(false, true), listed.map { it.folder })

        // The library already has part 2; the folder holds parts 1 and 2, a picture, and part 3 in a sub-folder.
        val root = temp.newFolder("books")
        PreparedAudio.import(root, stage("002 - Ba.opus" to "audio-2", "002 - Ba.opus.sync.json" to sync(2, "002 - Ba.opus", "Ba")))
        val content = mapOf(
            "a1" to ("001 - Một.opus" to "audio-1"), "s1" to ("001 - Một.opus.sync.json" to sync(1, "001 - Một.opus", "Một")),
            "a2" to ("002 - Ba.opus" to "audio-2"), "s2" to ("002 - Ba.opus.sync.json" to sync(2, "002 - Ba.opus", "Ba")),
            "p" to ("bia.jpg" to "x"), "a3" to ("003 - Năm.opus" to "audio-3"), "s3" to ("003 - Năm.opus.sync.json" to sync(3, "003 - Năm.opus", "Năm")))
        fun entries(vararg ids: String) = ids.map { PreparedAudio.DriveEntry(it, content.getValue(it).first, false) }
        val fetched = mutableListOf<String>()
        val counts = mutableListOf<Pair<Int, Int>>()
        val staging = temp.newFolder()
        val skipped = PreparedAudio.fetchDriveFolder(root, "top", staging, progress = { done, total -> counts += done to total },
            list = { if (it == "top") entries("a1", "s1", "a2", "s2", "p") + PreparedAudio.DriveEntry("sub", "phần sau", true) else entries("a3", "s3") },
            fetch = { entry, into -> fetched += entry.name; File(into, entry.name).apply { writeText(content.getValue(entry.id).second) } })
        assertEquals(1, skipped)
        assertEquals(listOf("001 - Một.opus.sync.json", "002 - Ba.opus.sync.json", "003 - Năm.opus.sync.json", "001 - Một.opus", "003 - Năm.opus"), fetched)
        assertEquals(5 to 5, counts.last())
        val result = PreparedAudio.import(root, staging)
        assertEquals(2, result.parts); assertEquals(0, result.withoutAudio)
        val book = json.decodeFromString(PreparedBook.serializer(), File(File(root, result.books.single()), "prepared.json").readText())
        assertEquals(listOf(1, 2, 3), book.parts.map { it.index })
        assertThrows(LinkFailed::class.java) { PreparedAudio.fetchDriveFolder(root, "empty", temp.newFolder(), list = { emptyList() }) }
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
