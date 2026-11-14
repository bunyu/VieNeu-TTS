package io.github.bunyu.reader

import io.github.bunyu.reader.text.Chunker
import io.github.bunyu.reader.text.Pause
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChunkerTest {
    @Test fun sentencesDoNotBreakOnAbbreviationsOrDecimals() {
        val text = "GS. Nam đến TP. Huế lúc 7h. Giá là 3.5 triệu. Anh hỏi: \"Thật sao?\" Cô gật đầu."
        assertEquals(listOf("GS. Nam đến TP. Huế lúc 7h.", "Giá là 3.5 triệu.", "Anh hỏi: \"Thật sao?\"", "Cô gật đầu."), Chunker.sentences(text))
    }

    @Test fun piecesStayWithinLimitAndMarkPauses() {
        val paragraph = (1..30).joinToString(" ") { "Đây là câu thứ $it trong đoạn văn dài." }
        val pieces = Chunker.split(paragraph + "\n\nĐoạn hai ngắn.", maxChars = 200)
        assertTrue(pieces.all { it.text.length <= 200 })
        assertEquals(Pause.PARAGRAPH, pieces[pieces.size - 2].pause)
        assertEquals(Pause.CHAPTER, pieces.last().pause)
        assertEquals("Đoạn hai ngắn.", pieces.last().text)
        // Nothing is lost or reordered.
        assertEquals((paragraph + " Đoạn hai ngắn.").replace(" ", ""), pieces.joinToString("") { it.text.replace(" ", "") })
    }

    @Test fun longSentencesBreakAtClausesNotInsideWords() {
        val sentence = (1..40).joinToString(", ") { "mệnh đề số $it" } + "."
        val parts = Chunker.fit(sentence, 120)
        assertTrue(parts.all { it.length <= 120 })
        assertTrue(parts.dropLast(1).all { it.endsWith(",") })
    }

    @Test fun quickStartSpeaksTheOpeningSentenceFirst() {
        val text = "Trời đã tối hẳn khi anh về đến nhà. Cô vẫn ngồi chờ bên cửa sổ, tay cầm cuốn sách cũ. Không ai nói với ai lời nào."
        val usual = Chunker.calls(text, maxChars = 140)
        assertEquals(Chunker.split(text, maxChars = 140, minChars = 40), usual)
        val quick = Chunker.calls(text, maxChars = 140, quickStart = true)
        assertEquals("Trời đã tối hẳn khi anh về đến nhà.", quick.first().text)
        assertEquals(Pause.SENTENCE, quick.first().pause)
        assertTrue(quick.all { it.text.length <= 140 })
        // The same words in the same order, only cut earlier.
        assertEquals(text, quick.joinToString(" ") { it.text })
    }

    @Test fun quickStartJoinsVeryShortOpeningSentences() {
        val quick = Chunker.calls("Ừ. Thế à? Anh không biết chuyện ấy đâu. Mọi người đều đã về từ chiều hôm qua, chỉ còn mình tôi ở lại trông nhà.", maxChars = 140, quickStart = true)
        assertEquals("Ừ. Thế à? Anh không biết chuyện ấy đâu.", quick.first().text)
    }

    @Test fun quickStartLeavesASingleSentenceAlone() {
        val text = "Chỉ có một câu ngắn thôi."
        assertEquals(Chunker.split(text, maxChars = 140, minChars = 40), Chunker.calls(text, maxChars = 140, quickStart = true))
    }
}
