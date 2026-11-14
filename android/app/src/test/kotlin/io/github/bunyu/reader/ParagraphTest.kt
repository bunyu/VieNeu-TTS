package io.github.bunyu.reader

import io.github.bunyu.reader.data.BookPiece
import io.github.bunyu.reader.data.OpenBook
import io.github.bunyu.reader.data.StoredBook
import io.github.bunyu.reader.data.StoredChapter
import io.github.bunyu.reader.text.Pause
import io.github.bunyu.reader.vieneu.NanoPresets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ParagraphTest {
    private val book = OpenBook(
        StoredBook("id", "Sách", null, listOf(StoredChapter("Một", ""), StoredChapter("Hai", ""))),
        listOf(
            BookPiece(0, 0, "Câu một.", Pause.SENTENCE),
            BookPiece(1, 0, "Câu hai.", Pause.PARAGRAPH),
            BookPiece(2, 0, "Đoạn sau chưa hết", Pause.SENTENCE),   // a chapter ends without a paragraph pause
            BookPiece(3, 1, "Hai", Pause.PARAGRAPH),
            BookPiece(4, 1, "Hết.", Pause.CHAPTER),
        ))

    @Test fun piecesJoinIntoParagraphsAndChaptersSplitThem() {
        assertEquals(listOf("Câu một. Câu hai.", "Đoạn sau chưa hết", "Hai", "Hết."), book.paragraphs.map { it.text })
        assertEquals(listOf(0, 0, 1, 2, 3), book.paragraphOf.toList())
        assertEquals(listOf(0, 2, 3, 4), book.paragraphs.map { it.first })
    }

    @Test fun aPieceIsFoundInsideItsParagraph() {
        val paragraph = book.paragraphs[0]
        assertEquals(0..7, paragraph.range(0))
        assertEquals("Câu hai.", paragraph.text.substring(paragraph.range(1)!!))
        assertNull(paragraph.range(2))
        assertEquals(0, paragraph.pieceAt(3).index)
        assertEquals(1, paragraph.pieceAt(9).index)
        assertEquals(1, paragraph.pieceAt(500).index)
    }

    @Test fun extraVoicesAreAddedAfterTheModelsOwnAndNeverReplaceThem() {
        fun voices(vararg names: String) = NanoPresets.parse("""{"default_voice":"${names[0]}","presets":{${names.joinToString(",") {
            """"$it":{"description":"$it","speaker_emb":[0.5,1],"style":[[1,2],[3,4]]}"""
        }}}}""")
        val all = voices("A", "B") + voices("B", "C")
        assertEquals(listOf("A", "B", "C"), all.voices.map { it.name })
        assertEquals("A", all.default?.name)
        assertEquals(2, all["C"]!!.voice.styleTokens)
    }
}
