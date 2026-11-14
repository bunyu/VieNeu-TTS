package io.github.bunyu.reader.playback

import java.io.File

/**
 * How many threads the model should use: one per fast core. A thread that lands on an efficiency core
 * finishes late and holds every step back, so those cores are left out.
 */
object BigCores {
    /** Fast cores of this phone, read from each core's top frequency; half the cores when that cannot be read. */
    fun count(): Int {
        val cores = Runtime.getRuntime().availableProcessors()
        val top = (0 until cores).mapNotNull { core ->
            runCatching { File("/sys/devices/system/cpu/cpu$core/cpufreq/cpuinfo_max_freq").readText().trim().toLong() }.getOrNull()
        }
        return if (top.size == cores) count(top) else (cores / 2).coerceIn(2, 4)
    }

    /** Cores within a quarter of the fastest one's top frequency, between 2 and 6. */
    internal fun count(topFrequencies: List<Long>): Int {
        val fastest = topFrequencies.maxOrNull() ?: return 2
        return topFrequencies.count { it >= fastest * 3 / 4 }.coerceIn(2, 6)
    }
}
