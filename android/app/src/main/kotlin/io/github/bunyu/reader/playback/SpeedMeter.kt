package io.github.bunyu.reader.playback

/**
 * How fast the voice is generated compared with listening: seconds of speech made per second of work,
 * over the last few model calls. Above 1 the phone keeps ahead of the listener; below 1 reading stalls
 * unless passages were prepared beforehand.
 */
class SpeedMeter(private val window: Int = 8) {
    private val calls = ArrayDeque<Pair<Double, Double>>()

    /** Records one model call and returns the speed over the window, or null while nothing was measured. */
    @Synchronized fun add(audioSeconds: Double, workSeconds: Double): Float? {
        if (audioSeconds > 0 && workSeconds > 0) {
            calls.addLast(audioSeconds to workSeconds)
            while (calls.size > window) calls.removeFirst()
        }
        return speed()
    }

    @Synchronized fun speed(): Float? {
        val work = calls.sumOf { it.second }
        return if (work > 0) (calls.sumOf { it.first } / work).toFloat() else null
    }

    @Synchronized fun reset() = calls.clear()
}
