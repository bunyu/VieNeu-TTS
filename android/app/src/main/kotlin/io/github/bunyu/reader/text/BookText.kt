package io.github.bunyu.reader.text

/** One chapter of a book: its heading and its text, paragraphs separated by new lines. */
data class ChapterText(val title: String, val text: String)

/** Reads at most [limit] bytes; null when the stream holds more. (InputStream.readNBytes needs API 33.) */
internal fun java.io.InputStream.readBounded(limit: Long): ByteArray? {
    val out = java.io.ByteArrayOutputStream()
    val buffer = ByteArray(65536)
    while (true) {
        val n = read(buffer)
        if (n < 0) break
        if (out.size() + n > limit) return null
        out.write(buffer, 0, n)
    }
    return out.toByteArray()
}
