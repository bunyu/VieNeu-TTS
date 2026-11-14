package io.github.bunyu.reader

import io.github.bunyu.reader.playback.BigCores
import org.junit.Assert.assertEquals
import org.junit.Test

class BigCoresTest {
    @Test fun onlyTheFastCoresCount() {
        // Dimensity 8300 (Xiaomi 14T): one 3.35 GHz and three 3.2 GHz cores, four 2.2 GHz efficiency cores.
        assertEquals(4, BigCores.count(listOf(2200000L, 2200000L, 2200000L, 2200000L, 3200000L, 3200000L, 3200000L, 3350000L)))
        // Eight equal cores are capped, and a phone with one fast core still gets two threads.
        assertEquals(6, BigCores.count(List(8) { 2000000L }))
        assertEquals(2, BigCores.count(listOf(1800000L, 1800000L, 1800000L, 3000000L)))
    }
}
