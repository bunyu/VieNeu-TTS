package io.github.bunyu.reader

import io.github.bunyu.reader.playback.SpeedMeter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SpeedMeterTest {
    @Test fun speedIsSpeechPerSecondOfWork() {
        val meter = SpeedMeter()
        assertNull(meter.speed())
        // 6 s of speech in 4 s of work, then 4 s in 1 s: 10 s in 5 s.
        assertEquals(1.5f, meter.add(6.0, 4.0)!!, 1e-6f)
        assertEquals(2f, meter.add(4.0, 1.0)!!, 1e-6f)
    }

    @Test fun onlyTheLastCallsCount() {
        val meter = SpeedMeter(window = 2)
        meter.add(1.0, 10.0)
        meter.add(2.0, 1.0)
        assertEquals(2f, meter.add(2.0, 1.0)!!, 1e-6f)
    }

    @Test fun emptyResultsAndResetAreIgnored() {
        val meter = SpeedMeter()
        assertNull(meter.add(0.0, 1.0))
        meter.add(3.0, 1.0)
        meter.reset()
        assertNull(meter.speed())
    }
}
