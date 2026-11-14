package io.github.bunyu.reader

import io.github.bunyu.reader.playback.endPause
import io.github.bunyu.reader.text.Pause
import io.github.bunyu.reader.text.PauseRules
import io.github.bunyu.reader.text.PauseText
import io.github.bunyu.reader.text.PauseText.Segment
import org.junit.Assert.assertEquals
import org.junit.Test

class PauseTextTest {
    @Test fun defaultRulesSplitOnlyAtTags() {
        val text = "Trời mưa. Gió thổi, lá rơi."
        assertEquals(listOf(Segment(text, null)), PauseText.segments(text, PauseRules()))
        assertEquals("", PauseRules().key)
    }

    @Test fun aChangedMarkPausesExactlyThere() {
        val rules = PauseRules().with(",", 1.2)
        assertEquals(listOf(Segment("Gió thổi,", 1200), Segment("lá rơi. Hết.", null)),
            PauseText.segments("Gió thổi, lá rơi. Hết.", rules))
    }

    @Test fun decimalsAbbreviationsAndInitialsDoNotSplit() {
        val rules = PauseRules().with(".", 2.0).with(",", 1.0)
        val text = "Giá 3.5 triệu, 1,5 kg ở Tp. Hồ Chí Minh của ông A. Nam."
        val segments = PauseText.segments(text, rules)
        assertEquals(listOf(Segment("Giá 3.5 triệu,", 1000), Segment("1,5 kg ở Tp. Hồ Chí Minh của ông A. Nam.", 2000)), segments)
    }

    @Test fun ellipsisIsOneMarkInBothSpellings() {
        val rules = PauseRules().with(PauseRules.ELLIPSIS, 1.5)
        assertEquals(listOf(Segment("Rồi...", 1500), Segment("im lặng…", 1500), Segment("Hết", null)),
            PauseText.segments("Rồi... im lặng… Hết", rules))
    }

    @Test fun closingQuotesStayWithTheirSentence() {
        val rules = PauseRules().with("?", 0.9)
        assertEquals(listOf(Segment("“Ai đó?”", 900), Segment("Không ai.", null)), PauseText.segments("“Ai đó?” Không ai.", rules))
    }

    @Test fun tagsAddExactPausesAndAreNotSpoken() {
        assertEquals(listOf(Segment("Một", 800), Segment("hai", 20_000), Segment("ba", null)),
            PauseText.segments("Một [0.8] hai [10][10] ba", PauseRules()))
        // At most ten seconds per tag; a comma decimal works too; a leading tag is silence on its own.
        assertEquals(listOf(Segment("", 10_000), Segment("A", 1500)), PauseText.segments("[25] A [1,5]", PauseRules()))
        // A changed mark followed by a tag stops for both.
        assertEquals(listOf(Segment("Xong.", 2600), Segment("Tiếp", null)), PauseText.segments("Xong. [2] Tiếp", PauseRules().with(".", 0.6)))
        assertEquals("Một hai ba", PauseText.stripTags("Một [0.8] hai [10][10] ba"))
    }

    @Test fun bracketedTextThatIsNotATagIsKept() {
        assertEquals(listOf(Segment("Xem [chú thích] và [12a].", null)), PauseText.segments("Xem [chú thích] và [12a].", PauseRules()))
    }

    @Test fun settingTheDefaultValueMeansNaturalAgain() {
        val rules = PauseRules().with(";", 0.7).with(";", 0.4)
        assertEquals(emptyMap<String, Double>(), rules.custom)
        assertEquals(0.4, rules.seconds(";"), 0.0)
        val custom = PauseRules().with(PauseRules.NEW_LINE, 2.0).with(",", 0.3 + 0.1)
        assertEquals(custom, PauseRules.decode(custom.encode()))
        assertEquals("nl=2.0|,=0.4", custom.key)
        assertEquals(PauseRules.MAX_SECONDS, PauseRules().with(".", 99.0).seconds("."), 0.0)
    }

    @Test fun pieceEndsUseTheNewLinePauseAtParagraphs() {
        val natural = PauseRules()
        assertEquals(Pause.PARAGRAPH.millis, endPause(Pause.PARAGRAPH, null, natural))
        assertEquals(Pause.SENTENCE.millis, endPause(Pause.SENTENCE, null, natural))
        val lines = natural.with(PauseRules.NEW_LINE, 2.0)
        assertEquals(2000, endPause(Pause.PARAGRAPH, 500, lines))
        assertEquals(500, endPause(Pause.SENTENCE, 500, lines))
        assertEquals(2000, endPause(Pause.CHAPTER, null, lines))
        assertEquals(Pause.CHAPTER.millis, endPause(Pause.CHAPTER, null, natural.with(PauseRules.NEW_LINE, 0.2)))
    }
}
