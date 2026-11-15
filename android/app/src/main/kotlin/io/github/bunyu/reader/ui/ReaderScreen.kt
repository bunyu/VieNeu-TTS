package io.github.bunyu.reader.ui

import android.app.Activity
import android.content.Intent
import android.view.WindowManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsIgnoringVisibility
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import io.github.bunyu.reader.R
import io.github.bunyu.reader.ReaderApp
import io.github.bunyu.reader.data.Bookmark
import io.github.bunyu.reader.data.OpenBook
import io.github.bunyu.reader.data.Paragraph
import io.github.bunyu.reader.data.ReaderSettings
import io.github.bunyu.reader.data.ReadingTheme
import io.github.bunyu.reader.data.Settings
import io.github.bunyu.reader.playback.NarrationState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.launch

/**
 * The book screen. It opens as a plain reader: the text in paragraphs, bars that hide on a tap, the
 * reader's own colours and type. The headphone button brings in the player, and with it the passage
 * being read aloud is highlighted and kept on screen.
 */
@Composable
fun ReaderScreen(bookId: String, onBack: () -> Unit, onSetup: (Tab) -> Unit) {
    val app = ReaderApp.get(LocalContext.current)
    val settings by app.settings.flow.collectAsState()
    MaterialTheme(colorScheme = readingColors(settings.theme)) { ReaderContent(bookId, settings, onBack, onSetup) }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
private fun ReaderContent(bookId: String, settings: Settings, onBack: () -> Unit, onSetup: (Tab) -> Unit) {
    val app = ReaderApp.get(LocalContext.current)
    val narration = app.narration
    val state by narration.state.collectAsState()
    val scope = rememberCoroutineScope()
    var dialog by remember { mutableStateOf<String?>(null) }
    var bars by rememberSaveable { mutableStateOf(true) }
    /** The player folded down to one row, to leave the page to the text while listening. */
    var compact by rememberSaveable { mutableStateOf(false) }
    var bookmarks by remember { mutableStateOf(emptyList<Bookmark>()) }
    LaunchedEffect(bookId) { narration.open(bookId); bookmarks = app.library.bookmarks(bookId) }
    val book = if (state.bookId == bookId) narration.current() else null
    val listening = settings.listening
    // The player needs its buttons, so the bars only hide while reading.
    val showBars = bars || listening
    ReadingWindow(settings, immersive = !showBars)

    val density = LocalDensity.current
    var topBar by remember { mutableIntStateOf(0) }
    var player by remember { mutableIntStateOf(0) }
    val list = remember(book) { LazyListState(book?.paragraphOf?.getOrNull(state.piece) ?: 0) }
    val here = book?.paragraphOf?.getOrNull(state.piece)
    val marked = remember(bookmarks, book) { bookmarks.mapNotNull { book?.paragraphOf?.getOrNull(it.piece) }.toSet() }

    fun jump(piece: Int) {
        val open = book ?: return
        if (listening) narration.seekPiece(piece)
        else { scope.launch { list.scrollToItem(open.paragraphOf.getOrElse(piece) { 0 }) }; narration.readAt(piece) }
    }
    fun saveBookmarks(next: List<Bookmark>) { bookmarks = next; scope.launch { app.library.saveBookmarks(bookId, next) } }
    fun toggleBookmark() {
        val open = book ?: return
        val paragraph = here?.let(open.paragraphs::getOrNull) ?: return
        saveBookmarks(if (here in marked) bookmarks.filterNot { open.paragraphOf.getOrNull(it.piece) == here }
            else (bookmarks + Bookmark(paragraph.first, paragraph.chapter, paragraph.text.take(90), System.currentTimeMillis())).sortedBy { it.piece })
    }

    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        if (book == null) LinearProgressIndicator(Modifier.fillMaxWidth().align(Alignment.Center).padding(horizontal = 48.dp))
        else {
            // Listening: keep the passage being read on screen.
            LaunchedEffect(state.piece, listening, book) {
                if (!listening) return@LaunchedEffect
                val at = book.paragraphOf.getOrNull(state.piece) ?: return@LaunchedEffect
                val view = list.layoutInfo
                val end = view.viewportEndOffset - view.afterContentPadding
                if (view.visibleItemsInfo.none { it.index == at && it.offset >= 0 && (it.offset + it.size <= end || it.offset < 8) }) list.animateScrollToItem(at)
            }
            // Reading: the place in the book is wherever the reader stopped scrolling.
            LaunchedEffect(list, listening) {
                if (listening) return@LaunchedEffect
                snapshotFlow { if (list.isScrollInProgress) -1 else list.firstVisibleItemIndex }.filter { it >= 0 }.distinctUntilChanged().collect { at ->
                    if (book.paragraphOf.getOrNull(state.piece) != at) book.paragraphs.getOrNull(at)?.let { narration.readAt(it.first) }
                }
            }
            val size = settings.fontSize
            val style = MaterialTheme.typography.bodyLarge.copy(
                fontFamily = if (settings.serif) FontFamily.Serif else FontFamily.SansSerif, fontSize = size.sp, lineHeight = (size * settings.lineSpacing).sp,
                textAlign = if (settings.justify) TextAlign.Justify else TextAlign.Start)
            val margin = settings.margin.dp
            Column(Modifier.fillMaxSize().then(if (listening) Modifier else Modifier.windowInsetsPadding(WindowInsets.systemBarsIgnoringVisibility))) {
                LazyColumn(state = list,
                    // Under the player the text must clear both bars; while reading the bars only pass over it for a moment.
                    contentPadding = if (listening) with(density) { PaddingValues(margin, topBar.toDp() + 12.dp, margin, player.toDp() + 24.dp) } else PaddingValues(margin, 16.dp, margin, 24.dp),
                    modifier = Modifier.weight(1f).fillMaxWidth().pointerInput(listening) { if (!listening) detectTapGestures { bars = !bars } }) {
                    itemsIndexed(book.paragraphs, key = { _, it -> it.first }) { at, paragraph ->
                        ParagraphText(paragraph, book.isHeading(paragraph), state.piece.takeIf { listening }, at in marked, style, (size * 0.6f).dp,
                            if (listening) { piece -> narration.seekPiece(piece) } else null)
                    }
                }
                if (!listening && state.totalPieces > 0) Text(progress(state, book), Modifier.fillMaxWidth().padding(vertical = 6.dp), textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f))
            }
        }

        AnimatedVisibility(showBars, Modifier.align(Alignment.TopCenter), enter = fadeIn(), exit = fadeOut()) {
            TopAppBar(
                modifier = Modifier.onSizeChanged { topBar = it.height },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surfaceContainer),
                title = {
                    Column {
                        Text(state.title, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.titleMedium)
                        Text(state.chapterTitle, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodySmall)
                    }
                },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) } },
                actions = {
                    IconButton(onClick = { dialog = "contents" }) { Icon(Icons.AutoMirrored.Filled.List, stringResource(R.string.chapters)) }
                    IconButton(onClick = { toggleBookmark() }) {
                        if (here in marked) Icon(painterResource(R.drawable.ic_bookmark), stringResource(R.string.bookmark_remove), tint = MaterialTheme.colorScheme.primary)
                        else Icon(painterResource(R.drawable.ic_bookmark_border), stringResource(R.string.bookmark_add))
                    }
                    IconButton(onClick = { dialog = "appearance" }) { Icon(painterResource(R.drawable.ic_text), stringResource(R.string.appearance)) }
                    IconButton(onClick = { app.settings.update { it.copy(listening = !it.listening) } }) {
                        Icon(painterResource(R.drawable.ic_headphones), stringResource(if (listening) R.string.listen_hide else R.string.listen),
                            tint = if (listening) MaterialTheme.colorScheme.primary else LocalContentColor.current)
                    }
                },
            )
        }

        if (listening) Surface(Modifier.align(Alignment.BottomCenter).onSizeChanged { player = it.height }, tonalElevation = 3.dp, color = MaterialTheme.colorScheme.surfaceContainer) {
            // The app draws edge to edge, so the controls must stay above the system navigation bar.
            Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 16.dp, vertical = 10.dp)) {
                state.error?.let { SetupBanner(it, onSetup) }
                if (compact) Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = { narration.toggle() }) {
                        Icon(painterResource(if (state.playing) R.drawable.ic_pause else R.drawable.ic_play), stringResource(if (state.playing) R.string.pause else R.string.play))
                    }
                    LinearProgressIndicator(progress = { (state.piece + 1f) / state.totalPieces.coerceAtLeast(1) }, modifier = Modifier.weight(1f).padding(horizontal = 8.dp))
                    IconButton(onClick = { compact = false }) { Icon(Icons.Filled.KeyboardArrowUp, stringResource(R.string.player_show)) }
                } else {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.weight(1f)) { Status(state.waiting || state.playing && state.readyAhead == 0, state.readyAhead, state.skipped, state.speed, state.buffered) }
                    IconButton(onClick = { compact = true }, modifier = Modifier.size(32.dp)) { Icon(Icons.Filled.KeyboardArrowDown, stringResource(R.string.player_hide)) }
                }
                // Slower than listening: reading will stall, and half the steps is the one setting that helps.
                if ((state.speed ?: 1f) < 1f && !settings.fast) TextButton(onClick = { app.settings.update { it.copy(fast = true) } }) { Text(stringResource(R.string.fast_suggest)) }
                if (state.totalPieces > 0) LinearProgressIndicator(progress = { (state.piece + 1f) / state.totalPieces }, modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = { narration.previous() }) { Icon(painterResource(R.drawable.ic_previous), stringResource(R.string.previous)) }
                    FilledIconButton(onClick = { narration.toggle() }, modifier = Modifier.size(64.dp), colors = IconButtonDefaults.filledIconButtonColors()) {
                        Icon(painterResource(if (state.playing) R.drawable.ic_pause else R.drawable.ic_play), stringResource(if (state.playing) R.string.pause else R.string.play), Modifier.size(36.dp))
                    }
                    IconButton(onClick = { narration.next() }) { Icon(painterResource(R.drawable.ic_next), stringResource(R.string.next)) }
                }
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    AssistChip(onClick = { dialog = "speed" }, label = { Text("%.2f×".format(settings.speed).replace(".00", "")) }, leadingIcon = { Icon(painterResource(R.drawable.ic_speed), null, Modifier.size(18.dp)) })
                    AssistChip(onClick = { dialog = "sleep" }, label = { SleepLabel(state.sleepAt, state.sleepAtChapterEnd) }, leadingIcon = { Icon(painterResource(R.drawable.ic_timer), null, Modifier.size(18.dp)) })
                    FilterChip(selected = settings.musicOn && settings.musicUri != null, onClick = { dialog = "music" }, label = { Text(stringResource(R.string.music)) }, leadingIcon = { Icon(painterResource(R.drawable.ic_music), null, Modifier.size(18.dp)) })
                    FilterChip(selected = settings.pauses.custom.isNotEmpty(), onClick = { dialog = "pauses" }, label = { Text(stringResource(R.string.pause_chip)) })
                    FilterChip(selected = state.preparingAll, onClick = { narration.prepareAll(!state.preparingAll) },
                        label = { Text(if (state.preparingAll) stringResource(R.string.preparing_all, state.preparedTotal, state.totalPieces - state.preparedFrom, state.preparedFromChapter + 1) else stringResource(R.string.prepare_all)) },
                        leadingIcon = { Icon(painterResource(R.drawable.ic_prepare), null, Modifier.size(18.dp)) })
                }
                }
            }
        }
    }

    when (dialog) {
        "contents" -> book?.let { open -> ContentsDialog(open, state.chapter, bookmarks, { jump(it); dialog = null }, { gone -> saveBookmarks(bookmarks - gone) }) { dialog = null } }
        "appearance" -> AppearanceSheet(settings, app.settings) { dialog = null }
        "speed" -> SpeedDialog(settings.speed, { s -> app.settings.update { it.copy(speed = s) } }) { dialog = null }
        "sleep" -> SleepDialog({ narration.sleepAfter(it) }, { narration.sleepAtChapterEnd() }) { dialog = null }
        "music" -> MusicDialog { dialog = null }
        "pauses" -> PauseDialog(settings.pauses, { p -> app.settings.update { it.copy(pauses = p) } }) { dialog = null }
    }
}

/** A chapter title that the book reads out as its own piece is shown as a heading. */
private fun OpenBook.isHeading(paragraph: Paragraph) =
    paragraph.pieces.size == 1 && paragraph.first == chapterStarts.getOrNull(paragraph.chapter) && paragraph.text == book.chapters[paragraph.chapter].title

@Composable
private fun progress(state: NarrationState, book: OpenBook) =
    stringResource(R.string.reading_progress, state.chapter + 1, book.book.chapters.size, (state.piece + 1) * 100 / state.totalPieces.coerceAtLeast(1))

/** Full screen, bar icon colours, brightness and keep-awake for the book screen; all put back on leaving it. */
@Composable
private fun ReadingWindow(settings: Settings, immersive: Boolean) {
    val view = LocalView.current
    val window = (view.context as? Activity)?.window ?: return
    val systemDark = LocalAppDark.current
    val dark = settings.theme.isDark(systemDark)
    DisposableEffect(immersive, dark, settings.brightness, settings.keepScreenOn) {
        val controller = WindowCompat.getInsetsController(window, view)
        fun brightness(value: Float) { window.attributes = window.attributes.also { it.screenBrightness = value } }
        controller.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        if (immersive) controller.hide(WindowInsetsCompat.Type.systemBars()) else controller.show(WindowInsetsCompat.Type.systemBars())
        controller.isAppearanceLightStatusBars = !dark
        controller.isAppearanceLightNavigationBars = !dark
        brightness(if (settings.brightness < 0f) WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE else settings.brightness.coerceAtLeast(0.01f))
        view.keepScreenOn = settings.keepScreenOn
        onDispose {
            controller.show(WindowInsetsCompat.Type.systemBars())
            controller.isAppearanceLightStatusBars = !systemDark
            controller.isAppearanceLightNavigationBars = !systemDark
            brightness(WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE)
            view.keepScreenOn = false
        }
    }
}

/**
 * One paragraph of the page. [current] is the piece being read aloud, shown highlighted; with [onPiece]
 * a tap picks the piece under the finger.
 */
@Composable
private fun ParagraphText(paragraph: Paragraph, heading: Boolean, current: Int?, marked: Boolean, style: TextStyle, gap: androidx.compose.ui.unit.Dp, onPiece: ((Int) -> Unit)?) {
    val highlight = MaterialTheme.colorScheme.primaryContainer
    val range = current?.let(paragraph::range)
    val text = remember(paragraph, range, highlight) {
        buildAnnotatedString { append(paragraph.text); if (range != null) addStyle(SpanStyle(background = highlight), range.first, range.last + 1) }
    }
    var layout by remember { mutableStateOf<TextLayoutResult?>(null) }
    val mark = MaterialTheme.colorScheme.primary
    Text(text, color = MaterialTheme.colorScheme.onBackground, onTextLayout = { layout = it },
        style = if (heading) style.copy(fontWeight = FontWeight.Bold, fontSize = style.fontSize * 1.25f, lineHeight = style.lineHeight * 1.25f, textAlign = TextAlign.Start) else style,
        modifier = Modifier.fillMaxWidth()
            .then(if (heading) Modifier.padding(top = gap) else Modifier)
            // A bookmarked paragraph carries a thin bar in the margin.
            .then(if (marked) Modifier.drawBehind { drawRect(mark, Offset(-8.dp.toPx(), 0f), Size(3.dp.toPx(), size.height - gap.toPx())) } else Modifier)
            .then(if (onPiece != null) Modifier.pointerInput(paragraph) { detectTapGestures { at -> layout?.let { onPiece(paragraph.pieceAt(it.getOffsetForPosition(at)).index) } } } else Modifier)
            .padding(bottom = gap))
}

@Composable
private fun ContentsDialog(book: OpenBook, chapter: Int, bookmarks: List<Bookmark>, onJump: (Int) -> Unit, onDelete: (Bookmark) -> Unit, onDismiss: () -> Unit) {
    var showBookmarks by remember { mutableStateOf(false) }
    AlertDialog(onDismissRequest = onDismiss, confirmButton = { TextButton(onDismiss) { Text(stringResource(R.string.close)) } },
        title = {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = !showBookmarks, onClick = { showBookmarks = false }, label = { Text(stringResource(R.string.chapters)) })
                FilterChip(selected = showBookmarks, onClick = { showBookmarks = true }, label = { Text(stringResource(R.string.bookmarks) + if (bookmarks.isEmpty()) "" else " (${bookmarks.size})") })
            }
        },
        text = {
            if (!showBookmarks) {
                val list = rememberLazyListState(initialFirstVisibleItemIndex = (chapter - 2).coerceAtLeast(0))
                LazyColumn(state = list) {
                    itemsIndexed(book.book.chapters) { i, item ->
                        Text(item.title, Modifier.fillMaxWidth().clickable { onJump(book.chapterStarts.getOrElse(i) { 0 }) }.padding(vertical = 12.dp),
                            color = if (i == chapter) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface)
                    }
                }
            } else if (bookmarks.isEmpty()) Text(stringResource(R.string.bookmarks_empty))
            else LazyColumn {
                itemsIndexed(bookmarks, key = { _, it -> it.piece }) { _, bookmark ->
                    Row(Modifier.fillMaxWidth().clickable { onJump(bookmark.piece.coerceIn(0, (book.pieces.size - 1).coerceAtLeast(0))) }, verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f).padding(vertical = 8.dp)) {
                            Text(book.book.chapters.getOrNull(bookmark.chapter)?.title.orEmpty(), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(bookmark.text, style = MaterialTheme.typography.bodyMedium, maxLines = 2, overflow = TextOverflow.Ellipsis)
                        }
                        IconButton(onClick = { onDelete(bookmark) }) { Icon(Icons.Filled.Delete, stringResource(R.string.bookmark_remove)) }
                    }
                }
            }
        })
}

/** Colours and type of the page. The sheet has no scrim, so every change shows on the text behind it at once. */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
private fun AppearanceSheet(settings: Settings, store: ReaderSettings, onDismiss: () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss, scrimColor = Color.Transparent) {
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 16.dp)) {
            Text(stringResource(R.string.theme), style = MaterialTheme.typography.labelLarge)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for ((theme, label) in listOf(ReadingTheme.AUTO to R.string.theme_auto, ReadingTheme.LIGHT to R.string.theme_light, ReadingTheme.SEPIA to R.string.theme_sepia,
                    ReadingTheme.DARK to R.string.theme_dark, ReadingTheme.BLACK to R.string.theme_black))
                    FilterChip(selected = settings.theme == theme, onClick = { store.update { it.copy(theme = theme) } }, label = { Text(stringResource(label)) })
            }
            Text(stringResource(R.string.font_size, settings.fontSize), style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 8.dp))
            Slider(settings.fontSize.toFloat(), { v -> store.update { it.copy(fontSize = Math.round(v)) } },
                valueRange = ReaderSettings.FONT_SIZES.first.toFloat()..ReaderSettings.FONT_SIZES.last.toFloat())
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = settings.serif, onClick = { store.update { it.copy(serif = true) } }, label = { Text(stringResource(R.string.font_serif), fontFamily = FontFamily.Serif) })
                FilterChip(selected = !settings.serif, onClick = { store.update { it.copy(serif = false) } }, label = { Text(stringResource(R.string.font_sans), fontFamily = FontFamily.SansSerif) })
            }
            Text(stringResource(R.string.line_spacing, settings.lineSpacing), style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 8.dp))
            Slider(settings.lineSpacing, { v -> store.update { it.copy(lineSpacing = Math.round(v * 10f) / 10f) } }, valueRange = 1.2f..2.2f)
            Text(stringResource(R.string.margins, settings.margin), style = MaterialTheme.typography.labelLarge)
            Slider(settings.margin.toFloat(), { v -> store.update { it.copy(margin = Math.round(v)) } },
                valueRange = ReaderSettings.MARGINS.first.toFloat()..ReaderSettings.MARGINS.last.toFloat())
            Toggle(R.string.justify, settings.justify) { on -> store.update { it.copy(justify = on) } }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.brightness), Modifier.weight(1f), style = MaterialTheme.typography.labelLarge)
                FilterChip(selected = settings.brightness < 0f, onClick = { store.update { it.copy(brightness = if (it.brightness < 0f) 0.5f else -1f) } }, label = { Text(stringResource(R.string.brightness_phone)) })
            }
            Slider(if (settings.brightness < 0f) 0.5f else settings.brightness, { v -> store.update { it.copy(brightness = v) } }, valueRange = 0.01f..1f, enabled = settings.brightness >= 0f)
            Toggle(R.string.keep_screen_on, settings.keepScreenOn) { on -> store.update { it.copy(keepScreenOn = on) } }
        }
    }
}

@Composable
private fun Toggle(label: Int, on: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clickable { onChange(!on) }, verticalAlignment = Alignment.CenterVertically) {
        Text(stringResource(label), Modifier.weight(1f))
        Switch(on, onChange)
    }
}

@Composable
private fun Status(waiting: Boolean, readyAhead: Int, skipped: Int, speed: Float?, buffered: Int?) {
    val text = when {
        buffered != null -> stringResource(R.string.status_buffering, buffered, io.github.bunyu.reader.playback.Narration.HOLD_PIECES)
        waiting -> stringResource(R.string.status_waiting)
        readyAhead > 0 -> stringResource(R.string.status_ready, readyAhead)
        else -> stringResource(R.string.status_idle)
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(text, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
        if (skipped > 0) Text(stringResource(R.string.status_skipped, skipped), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(end = 8.dp))
        // Below 1× the phone generates slower than the listener hears, so reading stalls without prepared passages.
        if (speed != null) Text(stringResource(R.string.status_speed, speed), style = MaterialTheme.typography.labelSmall,
            color = if (speed < 1f) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun SetupBanner(error: String, onSetup: (Tab) -> Unit) {
    val (message, tab) = when (error) {
        "voice_missing" -> R.string.error_voice_missing to Tab.VOICES
        "engine_missing" -> R.string.error_engine_missing to Tab.SETTINGS
        "runtime_failed" -> R.string.error_runtime to Tab.SETTINGS
        else -> R.string.status_problem to null
    }
    Card(Modifier.fillMaxWidth().padding(12.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(message), Modifier.weight(1f), color = MaterialTheme.colorScheme.onErrorContainer)
            if (tab != null) TextButton({ onSetup(tab) }) { Text(stringResource(R.string.fix)) }
        }
    }
}

@Composable
private fun SleepLabel(at: Long?, chapterEnd: Boolean) {
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(at) { while (at != null) { now = System.currentTimeMillis(); delay(15_000) } }
    Text(when {
        chapterEnd -> stringResource(R.string.sleep_chapter_short)
        at != null -> stringResource(R.string.sleep_minutes_left, ((at - now) / 60_000 + 1).coerceAtLeast(1))
        else -> stringResource(R.string.sleep)
    })
}

@Composable
private fun SpeedDialog(speed: Float, onChange: (Float) -> Unit, onDismiss: () -> Unit) {
    var value by remember { mutableFloatStateOf(speed) }
    AlertDialog(onDismissRequest = onDismiss, confirmButton = { TextButton(onDismiss) { Text(stringResource(R.string.close)) } },
        title = { Text(stringResource(R.string.speed_title, "%.2f×".format(value))) },
        text = {
            Column {
                Slider(value, { value = kotlin.math.round(it * 20f) / 20f; onChange(value) }, valueRange = 0.5f..2f, steps = 29)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (preset in listOf(0.8f, 1f, 1.15f, 1.3f, 1.5f)) FilterChip(selected = value == preset, onClick = { value = preset; onChange(preset) }, label = { Text("$preset×".replace(".0×", "×")) })
                }
                Spacer(Modifier.height(8.dp))
                Text(stringResource(R.string.speed_hint), style = MaterialTheme.typography.bodySmall)
            }
        })
}

@Composable
private fun SleepDialog(onMinutes: (Int?) -> Unit, onChapterEnd: () -> Unit, onDismiss: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss, confirmButton = { TextButton(onDismiss) { Text(stringResource(R.string.close)) } },
        title = { Text(stringResource(R.string.sleep)) },
        text = {
            Column {
                for (minutes in listOf(10, 15, 30, 45, 60, 90, 120)) Text(stringResource(R.string.sleep_after, minutes),
                    Modifier.fillMaxWidth().clickable { onMinutes(minutes); onDismiss() }.padding(vertical = 12.dp))
                Text(stringResource(R.string.sleep_chapter), Modifier.fillMaxWidth().clickable { onChapterEnd(); onDismiss() }.padding(vertical = 12.dp))
                Text(stringResource(R.string.sleep_off), Modifier.fillMaxWidth().clickable { onMinutes(null); onDismiss() }.padding(vertical = 12.dp), color = MaterialTheme.colorScheme.primary)
            }
        })
}

@Composable
private fun MusicDialog(onDismiss: () -> Unit) {
    val context = LocalContext.current
    val app = ReaderApp.get(context)
    val settings by app.settings.flow.collectAsState()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            runCatching { context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            app.settings.update { it.copy(musicUri = uri.toString(), musicName = displayName(context, uri).ifEmpty { null }, musicOn = true) }
        }
    }
    AlertDialog(onDismissRequest = onDismiss, confirmButton = { TextButton(onDismiss) { Text(stringResource(R.string.close)) } },
        title = { Text(stringResource(R.string.music)) },
        text = {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(settings.musicName ?: stringResource(R.string.music_none), Modifier.weight(1f), maxLines = 2, overflow = TextOverflow.Ellipsis)
                    Spacer(Modifier.width(8.dp))
                    Switch(settings.musicOn, { on -> app.settings.update { it.copy(musicOn = on) } }, enabled = settings.musicUri != null)
                }
                TextButton({ pick.launch(arrayOf("audio/*")) }) { Text(stringResource(R.string.music_choose)) }
                Text(stringResource(R.string.music_volume, (settings.musicVolume * 100).toInt()), style = MaterialTheme.typography.labelMedium)
                Slider(settings.musicVolume, { v -> app.settings.update { it.copy(musicVolume = v) } }, valueRange = 0f..0.6f)
                Text(stringResource(R.string.music_hint), style = MaterialTheme.typography.bodySmall)
            }
        })
}
