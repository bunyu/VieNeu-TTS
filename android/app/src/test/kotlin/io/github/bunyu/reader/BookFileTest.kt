package io.github.bunyu.reader

import io.github.bunyu.reader.books.BookFile
import io.github.bunyu.reader.books.BookUnreadable
import io.github.bunyu.reader.books.Xhtml
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class BookFileTest {
    @get:Rule val temporary = TemporaryFolder()

    private fun epub(vararg members: Pair<String, String>): File = temporary.newFile("book.epub").also { file ->
        ZipOutputStream(file.outputStream()).use { zip -> for ((name, text) in members) { zip.putNextEntry(ZipEntry(name)); zip.write(text.toByteArray()); zip.closeEntry() } }
    }

    private val container = "META-INF/container.xml" to """<?xml version="1.0"?><container><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"""
    private val opf = "OEBPS/content.opf" to """<package><metadata><dc:title>Tây Du Ký</dc:title><dc:creator>Ngô Thừa Ân</dc:creator></metadata>
        <manifest><item id="c2" href="Text/ch%202.xhtml" media-type="application/xhtml+xml"/><item id="c1" href='Text/ch1.xhtml'/><item id="css" href="s.css"/><item id="cover" href="cover.xhtml"/></manifest>
        <spine><itemref idref="cover"/><itemref idref="c1"/><itemref idref="c2"/></spine></package>"""

    @Test fun readsChaptersInSpineOrderWithHeadingsAndMetadata() {
        val book = BookFile.read(epub(container, opf,
            "OEBPS/cover.xhtml" to "<html><body><img src='c.jpg'/></body></html>",
            "OEBPS/Text/ch1.xhtml" to """<html><head><title>x</title><style>p{}</style></head><body><h2>Hồi 1</h2>
                <p>Đá nứt&nbsp;sinh &quot;khỉ&quot;<sup class="footnote"><a>1</a></sup>.</p><p>Dòng<br/>mới &#7899; &#x1EA1;</p><script>bad()</script></body></html>""",
            "OEBPS/Text/ch 2.xhtml" to "<html><body><p>Không có tiêu đề.</p></body></html>"), "book.epub")
        assertEquals("Tây Du Ký", book.title)
        assertEquals("Ngô Thừa Ân", book.author)
        assertEquals(listOf("Hồi 1", "3"), book.chapters.map { it.title })
        assertEquals("Hồi 1\nĐá nứt sinh \"khỉ\".\nDòng\nmới ớ ạ", book.chapters[0].text)
        assertEquals("Không có tiêu đề.", book.chapters[1].text)
    }

    @Test fun refusesFilesThatAreNotBooks() {
        assertThrows(BookUnreadable::class.java) { BookFile.read(temporary.newFile("x.epub").apply { writeText("not a zip") }, "x.epub") }
        assertThrows(BookUnreadable::class.java) { BookFile.read(epub(container, opf), "book.epub") }
        // A spine entry pointing outside the archive is skipped, not followed.
        val escaping = "OEBPS/content.opf" to """<package><manifest><item id="a" href="../../etc/passwd"/></manifest><spine><itemref idref="a"/></spine></package>"""
        assertThrows(BookUnreadable::class.java) { BookFile.read(temporary.newFolder().let { File(it, "e.epub") }.also { f ->
            ZipOutputStream(f.outputStream()).use { z -> for ((n, t) in listOf(container, escaping)) { z.putNextEntry(ZipEntry(n)); z.write(t.toByteArray()); z.closeEntry() } }
        }, "e.epub") }
    }

    @Test fun plainTextSplitsOnChapterLinesAndDetectsEncoding() {
        val file = temporary.newFile("truyen.txt").apply { writeText("Lời mở đầu.\n\nChương 1: Khởi đầu\nNội dung một.\n\nCHƯƠNG 2\nNội dung hai.\n") }
        val book = BookFile.read(file, "Truyện hay.txt")
        assertEquals(listOf("Truyện hay", "Chương 1: Khởi đầu", "CHƯƠNG 2"), book.chapters.map { it.title })
        assertEquals("Nội dung hai.", book.chapters[2].text)
        assertNull(book.author)
        assertEquals("Việt", BookFile.decode(byteArrayOf(0xFF.toByte(), 0xFE.toByte()) + "Việt".toByteArray(Charsets.UTF_16LE)))
        assertEquals("Việt", BookFile.decode(byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()) + "Việt".toByteArray()))
        // Not valid UTF-8: read as Windows-1258, where 0xE0 is "à".
        assertEquals("là", BookFile.decode(byteArrayOf('l'.code.toByte(), 0xE0.toByte())))
    }

    @Test fun normalizesArchivePaths() {
        assertEquals("OEBPS/Text/a.xhtml", BookFile.normalize("OEBPS/./Images/../Text/a.xhtml"))
        assertFalse(BookFile.normalize("../a") == "a")
        assertEquals("a & b", Xhtml.unescape("a &amp; b"))
        assertEquals("&bogus;", Xhtml.unescape("&bogus;"))
    }
}
