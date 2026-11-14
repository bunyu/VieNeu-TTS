package io.github.bunyu.reader.text

import java.util.Locale
import kotlin.math.roundToInt

/**
 * How long the narrator stops at each punctuation mark. A mark left at its default keeps the model's own,
 * natural flow; a mark the user changed stops for exactly that many seconds. [custom] holds only the
 * changed marks, in seconds.
 */
data class PauseRules(val custom: Map<String, Double> = emptyMap()) {
    fun seconds(mark: String): Double = custom[mark] ?: DEFAULTS.getValue(mark)
    fun millis(mark: String): Int? = custom[mark]?.let { (it * 1000).roundToInt() }

    /** Part of the audio cache key: only changed marks matter, so the default rules keep today's audio. */
    val key: String get() = custom.toSortedMap().entries.joinToString("|") { "${code(it.key)}=${format(it.value)}" }

    /** Sets [mark] to [seconds]; the default value means "natural" again. */
    fun with(mark: String, seconds: Double): PauseRules {
        val value = (seconds.coerceIn(0.0, MAX_SECONDS) * 10).roundToInt() / 10.0
        return PauseRules(if (value == DEFAULTS.getValue(mark)) custom - mark else custom + (mark to value))
    }

    fun encode(): String = key

    companion object {
        const val NEW_LINE = "\n"
        const val ELLIPSIS = "…"
        const val MAX_SECONDS = 10.0
        /** The marks the settings list, in order, with the pause the model makes by itself (shown as the default). */
        val DEFAULTS: Map<String, Double> = linkedMapOf(
            "." to 0.5, "," to 0.3, "?" to 0.5, "!" to 0.5, ":" to 0.4, ";" to 0.4, ELLIPSIS to 0.8, NEW_LINE to 1.0,
        )
        private fun code(mark: String) = if (mark == NEW_LINE) "nl" else mark
        private fun format(value: Double) = String.format(Locale.ROOT, "%.1f", value)

        fun decode(text: String?): PauseRules {
            if (text.isNullOrBlank()) return PauseRules()
            val custom = text.split('|').mapNotNull { entry ->
                val mark = entry.substringBefore('=').let { if (it == "nl") NEW_LINE else it }
                val value = entry.substringAfter('=', "").toDoubleOrNull()
                if (mark in DEFAULTS && value != null) mark to value.coerceIn(0.0, MAX_SECONDS) else null
            }.toMap()
            return PauseRules(custom)
        }
    }
}

/**
 * Splits text where a pause must be exact: after a mark whose pause the user changed, and at inline tags
 * like `[0.8]` or `[2]` (seconds, at most 10 each; `[10][10]` makes 20). Tags are removed from what is
 * spoken. A segment's [Segment.pauseMillis] is null when nothing exact follows it.
 */
object PauseText {
    data class Segment(val text: String, val pauseMillis: Int?)

    private val TAG = Regex("""\[\s*(\d{1,2}(?:[.,]\d{1,3})?)\s*]""")
    private const val CLOSING = "\"')]»”’"

    /** Removes `[n]` pause tags, for showing or speaking text without them. */
    fun stripTags(text: String): String = TAG.replace(text, " ").replace(Regex(" {2,}"), " ").trim()

    fun segments(text: String, rules: PauseRules): List<Segment> {
        val out = mutableListOf<Segment>()
        val current = StringBuilder()
        var pending: Int? = null
        fun flush(pause: Int?) {
            val spoken = current.toString().trim()
            if (spoken.isNotEmpty() || pause != null) out += Segment(spoken, pause)
            current.clear()
        }
        var i = 0
        while (i < text.length) {
            val tag = TAG.matchAt(text, i)
            if (tag != null) {
                val seconds = tag.groupValues[1].replace(',', '.').toDouble().coerceIn(0.0, PauseRules.MAX_SECONDS)
                pending = (pending ?: 0) + (seconds * 1000).roundToInt()
                i = tag.range.last + 1
                continue
            }
            val c = text[i]
            if (pending != null && !c.isWhitespace()) { flush(pending); pending = null }
            val ellipsis = text.startsWith("...", i)
            val mark = when {
                ellipsis -> PauseRules.ELLIPSIS
                c == '\n' -> PauseRules.NEW_LINE
                else -> c.toString()
            }
            val length = if (ellipsis) 3 else 1
            val exact = rules.millis(mark)
            if (exact != null && breaksHere(text, i, length, mark)) {
                current.append(text, i, i + length)
                var j = i + length
                while (j < text.length && text[j] in CLOSING) { current.append(text[j]); j++ }
                pending = (pending ?: 0) + exact
                i = j
                continue
            }
            current.append(text, i, i + length)
            i += length
        }
        flush(pending)
        return out
    }

    /** A mark ends a stretch of speech only when a space, a line end or a tag follows (not "3.5", "a,b" or "Tp.HCM"). */
    private fun breaksHere(text: String, at: Int, length: Int, mark: String): Boolean {
        if (mark == PauseRules.NEW_LINE) return true
        var j = at + length
        while (j < text.length && text[j] in CLOSING) j++
        if (j < text.length && !text[j].isWhitespace() && TAG.matchAt(text, j) == null) return false
        // A lone dot after an abbreviation or an initial does not end a sentence.
        if (mark == "." && !text.startsWith("..", at) && (at == 0 || text[at - 1] != '.')) {
            val word = text.substring(0, at).takeLastWhile { !it.isWhitespace() }.lowercase()
            if (word in Chunker.NO_BREAK || (word.length == 1 && word[0].isLetter())) return false
        }
        // The dots of "..." count as one ellipsis, handled where the first dot is.
        if (mark == "." && ((at > 0 && text[at - 1] == '.') || text.startsWith("..", at))) return false
        return true
    }
}
