package io.github.bunyu.reader.data

import android.content.Context
import io.github.bunyu.reader.text.Dialect
import io.github.bunyu.reader.text.PauseRules
import io.github.bunyu.reader.text.VietnameseText
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Which voice reads: one of the model's own Vietnamese voices, or a voice the user cloned from a recording. */
sealed interface VoiceChoice {
    val key: String
    data class Cloned(val id: String) : VoiceChoice { override val key get() = "clone:$id" }
    data class Preset(val name: String) : VoiceChoice { override val key get() = "nano:$name" }
    companion object {
        // Choices saved by the first version of the app ("preset:3") name voices of engines it no longer has; they fall back to the default.
        fun parse(value: String?): VoiceChoice? = when {
            value == null -> null
            value.startsWith("clone:") -> Cloned(value.removePrefix("clone:"))
            value.startsWith("nano:") -> value.removePrefix("nano:").takeIf { it.isNotBlank() && it.length <= 80 }?.let(::Preset)
            else -> null
        }
    }
}

/** The paper and ink of the reading screen; AUTO follows the light or dark mode of the phone. */
enum class ReadingTheme { AUTO, LIGHT, SEPIA, DARK, BLACK }

data class Settings(
    val voice: VoiceChoice? = null,
    val speed: Float = 1f,
    val musicUri: String? = null,
    val musicName: String? = null,
    val musicVolume: Float = 0.15f,
    val musicOn: Boolean = false,
    val dialect: Dialect = Dialect.NORTH,
    val pronunciations: String = "",
    val allowMobileData: Boolean = false,
    /** How many pieces (a few sentences each) to prepare ahead of the listening position. */
    val prepareAhead: Int = 40,
    /** Half the generation steps: about twice as fast, slightly rougher. */
    val fast: Boolean = false,
    /** Pauses the user set per punctuation mark; unchanged marks keep the model's natural flow. */
    val pauses: PauseRules = PauseRules(),
    /** The book screen shows the player and prepares audio only while this is on; off, it is a plain reader. */
    val listening: Boolean = false,
    val theme: ReadingTheme = ReadingTheme.AUTO,
    /** Text size in sp. */
    val fontSize: Int = 19,
    val serif: Boolean = true,
    /** Line height as a multiple of the text size. */
    val lineSpacing: Float = 1.6f,
    /** Side margins in dp. */
    val margin: Int = 20,
    val justify: Boolean = false,
    /** Screen brightness while reading, 0.01 to 1; negative follows the phone. */
    val brightness: Float = -1f,
    val keepScreenOn: Boolean = true,
) {
    fun textRules() = VietnameseText(dialect, VietnameseText.parsePronunciations(pronunciations))
}

class ReaderSettings(context: Context) {
    private val prefs = context.getSharedPreferences("reader", Context.MODE_PRIVATE)
    private val state = MutableStateFlow(load())
    val flow: StateFlow<Settings> = state.asStateFlow()
    val value get() = state.value

    private fun load() = Settings(
        voice = VoiceChoice.parse(prefs.getString("voice", null)),
        speed = prefs.getFloat("speed", 1f).coerceIn(0.5f, 2f),
        musicUri = prefs.getString("musicUri", null),
        musicName = prefs.getString("musicName", null),
        musicVolume = prefs.getFloat("musicVolume", 0.15f).coerceIn(0f, 1f),
        musicOn = prefs.getBoolean("musicOn", false),
        dialect = runCatching { Dialect.valueOf(prefs.getString("dialect", null) ?: "NORTH") }.getOrDefault(Dialect.NORTH),
        pronunciations = prefs.getString("pronunciations", "").orEmpty(),
        allowMobileData = prefs.getBoolean("allowMobileData", false),
        prepareAhead = prefs.getInt("prepareAhead", 40).coerceIn(5, 400),
        fast = prefs.getBoolean("fast", false),
        pauses = PauseRules.decode(prefs.getString("pauses", null)),
        listening = prefs.getBoolean("listening", false),
        theme = runCatching { ReadingTheme.valueOf(prefs.getString("theme", null) ?: "AUTO") }.getOrDefault(ReadingTheme.AUTO),
        fontSize = prefs.getInt("fontSize", 19).coerceIn(FONT_SIZES.first, FONT_SIZES.last),
        serif = prefs.getBoolean("serif", true),
        lineSpacing = prefs.getFloat("lineSpacing", 1.6f).coerceIn(1.2f, 2.2f),
        margin = prefs.getInt("margin", 20).coerceIn(MARGINS.first, MARGINS.last),
        justify = prefs.getBoolean("justify", false),
        brightness = prefs.getFloat("brightness", -1f).coerceIn(-1f, 1f),
        keepScreenOn = prefs.getBoolean("keepScreenOn", true),
    )

    fun update(change: (Settings) -> Settings) {
        val next = change(state.value)
        prefs.edit()
            .putString("voice", next.voice?.key)
            .putFloat("speed", next.speed.coerceIn(0.5f, 2f))
            .putString("musicUri", next.musicUri)
            .putString("musicName", next.musicName)
            .putFloat("musicVolume", next.musicVolume.coerceIn(0f, 1f))
            .putBoolean("musicOn", next.musicOn)
            .putString("dialect", next.dialect.name)
            .putString("pronunciations", next.pronunciations.take(50_000))
            .putBoolean("allowMobileData", next.allowMobileData)
            .putInt("prepareAhead", next.prepareAhead.coerceIn(5, 400))
            .putBoolean("fast", next.fast)
            .putString("pauses", next.pauses.encode())
            .putBoolean("listening", next.listening)
            .putString("theme", next.theme.name)
            .putInt("fontSize", next.fontSize.coerceIn(FONT_SIZES.first, FONT_SIZES.last))
            .putBoolean("serif", next.serif)
            .putFloat("lineSpacing", next.lineSpacing.coerceIn(1.2f, 2.2f))
            .putInt("margin", next.margin.coerceIn(MARGINS.first, MARGINS.last))
            .putBoolean("justify", next.justify)
            .putFloat("brightness", next.brightness.coerceIn(-1f, 1f))
            .putBoolean("keepScreenOn", next.keepScreenOn)
            .apply()
        state.value = next
    }

    companion object {
        val FONT_SIZES = 14..32
        val MARGINS = 8..40
    }
}
