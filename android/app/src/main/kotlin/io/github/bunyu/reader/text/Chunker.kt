package io.github.bunyu.reader.text

/** How long the narrator pauses after a piece, so sentences and paragraphs breathe like a human reading. */
enum class Pause(val millis: Int) { SENTENCE(250), PARAGRAPH(650), CHAPTER(1400) }

/** One piece of a chapter, read in a single model call: [text] is shown to the reader, [pause] follows it. */
data class Piece(val text: String, val pause: Pause)

/**
 * Splits chapter text into pieces of a few sentences. Each model call clones the voice afresh, so
 * pieces are long enough to keep the delivery steady but short enough to start playing quickly and to
 * stay inside what the model reads reliably. Sentences never break on abbreviations or decimals.
 */
object Chunker {
    /** Abbreviations whose dot does not end a sentence. Compared lowercase, without the dot. */
    internal val NO_BREAK = setOf("tp", "gs", "pgs", "ts", "ths", "bs", "ks", "v.v", "mr", "mrs", "ms", "dr", "st", "tr", "q", "p", "ông", "bà", "no", "vol")
    private val END = Regex("[.!?…]+[\"')\\]»”’]*")

    fun split(text: String, maxChars: Int = 280, minChars: Int = 60): List<Piece> {
        require(maxChars in 40..2000 && minChars in 1 until maxChars)
        val paragraphs = text.split(Regex("\\n\\s*")).map { it.trim() }.filter { it.isNotEmpty() && it.any(Char::isLetterOrDigit) }
        val pieces = mutableListOf<Piece>()
        for (paragraph in paragraphs) {
            val sentences = sentences(paragraph).flatMap { fit(it, maxChars) }
            var current = StringBuilder()
            fun flush(pause: Pause) {
                if (current.isNotBlank()) pieces += Piece(current.toString().trim(), pause)
                current = StringBuilder()
            }
            for (sentence in sentences) {
                if (current.isNotEmpty() && current.length + 1 + sentence.length > maxChars && current.length >= minChars) flush(Pause.SENTENCE)
                if (current.isNotEmpty()) current.append(' ')
                current.append(sentence)
            }
            flush(Pause.PARAGRAPH)
        }
        if (pieces.isNotEmpty()) pieces[pieces.lastIndex] = pieces.last().copy(pause = Pause.CHAPTER)
        return pieces
    }

    /**
     * The model calls for one spoken stretch of a piece. With [quickStart] the calls start small and grow:
     * the opening sentences (at least [quickChars] characters), then about [rampChars], then the usual
     * size. A listener who is waiting hears the first words sooner, and because each call is generated
     * while the one before it plays, no call may be much longer than what is already playing.
     */
    fun calls(spoken: String, maxChars: Int, minChars: Int = 40, quickStart: Boolean = false, quickChars: Int = 30, rampChars: Int = 70): List<Piece> {
        val whole = split(spoken, maxChars, minChars)
        val text = spoken.trim()
        if (!quickStart || whole.isEmpty() || '\n' in text) return whole
        val sentences = sentences(text)
        var used = 0
        /** Whole sentences until [atLeast] characters are reached, never past [atMost] once there is one. */
        fun take(atLeast: Int, atMost: Int): String {
            var taken = ""
            while (used < sentences.size) {
                val next = sentences[used]
                if (taken.isNotEmpty() && (taken.length >= atLeast || taken.length + 1 + next.length > atMost)) break
                taken = if (taken.isEmpty()) next else "$taken $next"
                used++
            }
            return taken
        }
        val head = take(quickChars, maxChars)
        // No gain when the opening is the whole stretch, or as long as the usual first call.
        if (used == sentences.size || head.length > maxChars || head.length >= whole.first().text.length) return whole
        val second = take(rampChars, rampChars)
        val rest = sentences.drop(used).joinToString(" ")
        return listOf(Piece(head, Pause.SENTENCE)) + split(second, maxChars, minChars).map { it.copy(pause = Pause.SENTENCE) } +
            if (rest.isEmpty()) emptyList() else split(rest, maxChars, minChars)
    }

    internal fun sentences(paragraph: String): List<String> {
        val result = mutableListOf<String>()
        var start = 0
        for (match in END.findAll(paragraph)) {
            val end = match.range.last + 1
            if (end >= paragraph.length) break
            if (!paragraph[end].isWhitespace()) continue
            val next = paragraph.substring(end).trimStart().firstOrNull() ?: break
            if (!(next.isUpperCase() || next.isDigit() || next in "\"'“‘(«-–—")) continue
            if (match.value.startsWith(".") && match.value.length == 1) {
                val word = paragraph.substring(start, match.range.first).takeLastWhile { !it.isWhitespace() }.lowercase()
                if (word in NO_BREAK || (word.length == 1 && word[0].isLetter())) continue
            }
            result += paragraph.substring(start, end).trim()
            start = end
        }
        if (start < paragraph.length) result += paragraph.substring(start).trim()
        return result.filter { it.isNotEmpty() }
    }

    /** Breaks an over-long sentence at clause marks, then at spaces, never inside a word. */
    internal fun fit(sentence: String, maxChars: Int): List<String> {
        if (sentence.length <= maxChars) return listOf(sentence)
        val out = mutableListOf<String>()
        var rest = sentence
        while (rest.length > maxChars) {
            val window = rest.substring(0, maxChars)
            val clause = window.indexOfLast { it in ",;:" }.takeIf { it >= maxChars / 3 }
            val space = window.lastIndexOf(' ').takeIf { it >= maxChars / 3 }
            val cut = (clause ?: space ?: (maxChars - 1)) + 1
            out += rest.substring(0, cut).trim()
            rest = rest.substring(cut).trim()
        }
        if (rest.isNotEmpty()) out += rest
        return out
    }
}
