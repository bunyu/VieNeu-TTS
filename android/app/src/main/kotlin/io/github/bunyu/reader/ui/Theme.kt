package io.github.bunyu.reader.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import io.github.bunyu.reader.data.ReadingTheme
import io.github.bunyu.reader.data.AppTheme
import android.app.Activity
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.sp

/** Warm paper tones for long reading sessions; a dark scheme for reading in bed. */
private val Light = lightColorScheme(
    primary = Color(0xFFB4552D), onPrimary = Color.White, primaryContainer = Color(0xFFFFDBCD), onPrimaryContainer = Color(0xFF3A0B00),
    secondary = Color(0xFF77574A), secondaryContainer = Color(0xFFFFDBCD),
    background = Color(0xFFFBF4EA), onBackground = Color(0xFF2B2118), surface = Color(0xFFFBF4EA), onSurface = Color(0xFF2B2118),
    surfaceVariant = Color(0xFFF1E3D3), onSurfaceVariant = Color(0xFF53433A), surfaceContainer = Color(0xFFF4E9DC),
)
private val Dark = darkColorScheme(
    primary = Color(0xFFFFB597), onPrimary = Color(0xFF5E1700), primaryContainer = Color(0xFF8A3A16), onPrimaryContainer = Color(0xFFFFDBCD),
    background = Color(0xFF17120E), onBackground = Color(0xFFEDE0D4), surface = Color(0xFF17120E), onSurface = Color(0xFFEDE0D4),
    surfaceVariant = Color(0xFF3A2F28), onSurfaceVariant = Color(0xFFD8C2B6), surfaceContainer = Color(0xFF231C17),
)

/** Whether the app is dark right now: the user's choice, or the phone's when left on automatic. */
val LocalAppDark = staticCompositionLocalOf { false }

/** Colours of the book screen for the reading theme the user chose. */
@Composable
fun readingColors(theme: ReadingTheme): ColorScheme = when (theme) {
    ReadingTheme.AUTO -> if (LocalAppDark.current) Dark else Light
    ReadingTheme.LIGHT -> Light.copy(background = Color(0xFFFDFDFB), surface = Color(0xFFFDFDFB), onBackground = Color(0xFF1C1B1A), onSurface = Color(0xFF1C1B1A))
    ReadingTheme.SEPIA -> Light.copy(background = Color(0xFFF1E2C6), surface = Color(0xFFF1E2C6), onBackground = Color(0xFF4A3622), onSurface = Color(0xFF4A3622),
        surfaceVariant = Color(0xFFE6D3B1), surfaceContainer = Color(0xFFEAD9B9))
    ReadingTheme.DARK -> Dark
    ReadingTheme.BLACK -> Dark.copy(background = Color.Black, surface = Color.Black, onBackground = Color(0xFFB8B0A8), onSurface = Color(0xFFB8B0A8),
        surfaceContainer = Color(0xFF121212))
}

fun ReadingTheme.isDark(systemDark: Boolean) = when (this) {
    ReadingTheme.AUTO -> systemDark
    ReadingTheme.DARK, ReadingTheme.BLACK -> true
    else -> false
}

@Composable
fun ReaderTheme(theme: AppTheme = AppTheme.AUTO, content: @Composable () -> Unit) {
    val base = Typography()
    val dark = when (theme) { AppTheme.AUTO -> isSystemInDarkTheme(); AppTheme.DARK -> true; AppTheme.LIGHT -> false }
    // The status and navigation bar icons must contrast with the app's colours, not the phone's mode.
    val view = LocalView.current
    val window = (view.context as? Activity)?.window
    if (window != null) SideEffect {
        WindowCompat.getInsetsController(window, view).apply { isAppearanceLightStatusBars = !dark; isAppearanceLightNavigationBars = !dark }
    }
    CompositionLocalProvider(LocalAppDark provides dark) {
        MaterialTheme(
            colorScheme = if (dark) Dark else Light,
            typography = base.copy(bodyLarge = TextStyle(fontFamily = FontFamily.Serif, fontSize = 19.sp, lineHeight = 30.sp)),
            content = content,
        )
    }
}
