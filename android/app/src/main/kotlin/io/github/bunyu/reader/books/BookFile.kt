package io.github.bunyu.reader.books

import io.github.bunyu.reader.text.ChapterText
import io.github.bunyu.reader.text.readBounded
import java.io.File
import java.net.URLDecoder
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.util.zip.ZipFile

/** A book's text, split into chapters, with whatever title and author the file declares. */
class BookContent(val chapters: List<ChapterText>, val title: String?, val author: String?)

class BookUnreadable(reason: String) : Exception(reason)

/**
 * Reads EPUB (2 and 3) and plain-text books. EPUB chapters follow the reading order (spine) of the package
 * document; each document's first heading names it. Images, styles and scripts are ignored. Archives are
 * read with fixed limits so a malformed or hostile file cannot exhaust memory.
 */
object BookFile {
    private const val MAX_FILE = 64L * 1024 * 1024
    private const val MAX_MEMBER = 16L * 1024 * 1024
    private const val MAX_TEXT = 4_000_000

    fun read(source: File, fileName: String): BookContent {
        if (source.length() > MAX_FILE) throw BookUnreadable("too_large")
        return when (fileName.substringAfterLast('.', "").lowercase()) {
            "txt", "md", "text" -> text(source, fileName.substringBeforeLast('.'))
            else -> epub(source)
        }
    }

    private fun text(source: File, name: String): BookContent {
        val bytes = source.readBytes()
        val text = decode(bytes).replace("\r\n", "\n").replace('\r', '\n').trim()
        if (text.none { it.isLetterOrDigit() }) throw BookUnreadable("no_text")
        // Lines such as "Chương 12" or "Chapter 3: ..." start chapters; without any, the file is one chapter.
        val heading = Regex("^(?:chương|chuong|hồi|quyển|phần|chapter|part)\\s+[\\p{L}\\p{N}]+.{0,80}$", RegexOption.IGNORE_CASE)
        val chapters = mutableListOf<ChapterText>()
        var title = name
        val body = StringBuilder()
        fun flush() { val t = body.toString().trim(); if (t.any { it.isLetterOrDigit() }) chapters += ChapterText(title, t); body.clear() }
        for (line in text.lines()) {
            if (heading.matches(line.trim())) { flush(); title = line.trim() } else body.append(line).append('\n')
        }
        flush()
        return BookContent(chapters.take(5000), name, null)
    }

    /** UTF-8 (with or without BOM), then UTF-16 by BOM, then Windows-1258 (older Vietnamese files). */
    internal fun decode(bytes: ByteArray): String {
        if (bytes.size >= 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xFE.toByte()) return String(bytes, 2, bytes.size - 2, Charsets.UTF_16LE)
        if (bytes.size >= 2 && bytes[0] == 0xFE.toByte() && bytes[1] == 0xFF.toByte()) return String(bytes, 2, bytes.size - 2, Charsets.UTF_16BE)
        val start = if (bytes.size >= 3 && bytes[0] == 0xEF.toByte() && bytes[1] == 0xBB.toByte() && bytes[2] == 0xBF.toByte()) 3 else 0
        return try {
            Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes, start, bytes.size - start)).toString()
        } catch (e: CharacterCodingException) {
            String(bytes, charset("windows-1258"))
        }
    }

    private fun epub(source: File): BookContent {
        val zip = try { ZipFile(source) } catch (e: Exception) { throw BookUnreadable("not_a_book") }
        zip.use {
            val entries = zip.entries().asSequence().take(10_001).toList()
            if (entries.size > 10_000) throw BookUnreadable("not_a_book")
            val byName = entries.filter { !it.isDirectory }.associateBy { normalize(it.name) }
            fun member(path: String): String? {
                val entry = byName[normalize(path)] ?: return null
                if (entry.size > MAX_MEMBER) return null
                return zip.getInputStream(entry).use { it.readBounded(MAX_MEMBER) }?.let(::decode)
            }
            val container = member("META-INF/container.xml") ?: throw BookUnreadable("not_a_book")
            val opfPath = attribute(Regex("<(?:\\w+:)?rootfile\\b[^>]*>").find(container)?.value ?: "", "full-path") ?: throw BookUnreadable("not_a_book")
            val opf = member(opfPath) ?: throw BookUnreadable("not_a_book")
            val base = normalize(opfPath).substringBeforeLast('/', "")
            val manifest = Regex("<(?:\\w+:)?item\\b[^>]*>").findAll(opf).mapNotNull { tag ->
                val id = attribute(tag.value, "id") ?: return@mapNotNull null
                val href = attribute(tag.value, "href") ?: return@mapNotNull null
                id to href
            }.toMap()
            val spine = Regex("<(?:\\w+:)?itemref\\b[^>]*>").findAll(opf).mapNotNull { attribute(it.value, "idref") }.toList()
            val chapters = mutableListOf<ChapterText>()
            var total = 0
            for ((index, id) in spine.withIndex()) {
                val href = manifest[id] ?: continue
                val relative = runCatching { URLDecoder.decode(href.substringBefore('#').replace("+", "%2B"), "UTF-8") }.getOrNull() ?: continue
                val document = member(if (base.isEmpty()) relative else "$base/$relative") ?: continue
                val (heading, text) = Xhtml.text(document)
                if (text.none { it.isLetterOrDigit() }) continue
                total += text.length
                if (total > MAX_TEXT) break
                chapters += ChapterText(heading ?: (index + 1).toString(), text)
            }
            if (chapters.isEmpty()) throw BookUnreadable("no_text")
            fun meta(name: String) = Regex("<dc:$name\\b[^>]*>([\\s\\S]*?)</dc:$name>").find(opf)?.groupValues?.get(1)
                ?.let(Xhtml::inline)?.takeIf { it.isNotEmpty() && it.length <= 300 }
            return BookContent(chapters, meta("title"), meta("creator"))
        }
    }

    private fun attribute(tag: String, name: String): String? =
        Regex("(?:^|\\s)$name\\s*=\\s*(\"([^\"]*)\"|'([^']*)')").find(tag)?.let { m -> Xhtml.unescape(m.groupValues[2].ifEmpty { m.groupValues[3] }) }

    /** Resolves "." and ".." so "OEBPS/../x" and "x" name the same member; a path that leaves the archive resolves to nothing. */
    internal fun normalize(path: String): String {
        val parts = ArrayList<String>()
        for (part in path.replace('\\', '/').split('/')) when (part) {
            "", "." -> {}
            ".." -> if (parts.isNotEmpty()) parts.removeAt(parts.lastIndex) else return "\u0000"
            else -> parts += part
        }
        return parts.joinToString("/")
    }
}

/** Readable text from (X)HTML: paragraphs and line breaks become new lines, everything else is dropped. */
object Xhtml {
    private val DROP = Regex("<(script|style|head|svg|math|rt|rp)\\b[\\s\\S]*?</\\1\\s*>", RegexOption.IGNORE_CASE)
    private val COMMENT = Regex("<!--[\\s\\S]*?-->|<!\\[CDATA\\[[\\s\\S]*?]]>|<\\?[\\s\\S]*?\\?>|<!DOCTYPE[^>]*>", RegexOption.IGNORE_CASE)
    private val BLOCK = Regex("</?(p|div|br|h[1-6]|li|tr|blockquote|section|article|header|footer|aside|hr|dt|dd|figcaption|pre|table)\\b[^>]*>", RegexOption.IGNORE_CASE)
    private val NOTE = Regex("<(sup|a)\\b[^>]*(?:epub:type\\s*=\\s*[\"']noteref[\"']|class\\s*=\\s*[\"'][^\"']*(?:footnote|noteref)[^\"']*[\"'])[^>]*>[\\s\\S]*?</\\1>", RegexOption.IGNORE_CASE)
    private val HEADING = Regex("<h([1-3])\\b[^>]*>([\\s\\S]*?)</h\\1\\s*>", RegexOption.IGNORE_CASE)
    private val TITLE = Regex("<title\\b[^>]*>([\\s\\S]*?)</title>", RegexOption.IGNORE_CASE)
    private val TAG = Regex("<[^>]*>")

    /** The document's first heading (or its title) and its body text. */
    fun text(document: String): Pair<String?, String> {
        val clean = COMMENT.replace(document, "")
        val heading = (HEADING.find(clean)?.groupValues?.get(2) ?: TITLE.find(clean)?.groupValues?.get(1))?.let(::inline)?.takeIf { it.isNotEmpty() && it.length <= 200 }
        val body = Regex("<body\\b[^>]*>([\\s\\S]*)</body>", RegexOption.IGNORE_CASE).find(clean)?.groupValues?.get(1) ?: clean
        var text = DROP.replace(body, " ")
        text = NOTE.replace(text, "")
        text = BLOCK.replace(text, "\n")
        text = unescape(TAG.replace(text, ""))
        val lines = text.lines().map { it.replace(Regex("[\\s\\u00A0]+"), " ").trim() }.filter { it.isNotEmpty() }
        return heading to lines.joinToString("\n")
    }

    fun inline(fragment: String) = unescape(TAG.replace(fragment, " ")).replace(Regex("[\\s\\u00A0]+"), " ").trim()

    private val NAMED = mapOf("amp" to "&", "lt" to "<", "gt" to ">", "quot" to "\"", "apos" to "'", "nbsp" to "\u00A0", "hellip" to "…",
        "mdash" to "—", "ndash" to "–", "lsquo" to "‘", "rsquo" to "’", "ldquo" to "“", "rdquo" to "”", "laquo" to "«", "raquo" to "»", "shy" to "", "zwnj" to "", "zwj" to "")

    fun unescape(text: String): String = Regex("&(#x[0-9a-fA-F]{1,6}|#[0-9]{1,7}|[a-zA-Z]{2,8});").replace(text) { m ->
        val code = m.groupValues[1]
        when {
            code.startsWith("#x") || code.startsWith("#X") -> code.substring(2).toIntOrNull(16)?.takeIf { Character.isValidCodePoint(it) }?.let { String(Character.toChars(it)) }
            code.startsWith("#") -> code.substring(1).toIntOrNull()?.takeIf { Character.isValidCodePoint(it) }?.let { String(Character.toChars(it)) }
            else -> NAMED[code]
        } ?: m.value
    }
}
