package io.github.bunyu.reader.playback

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.media3.common.AudioAttributes
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.extractor.DefaultExtractorsFactory
import androidx.media3.extractor.mp3.Mp3Extractor
import io.github.bunyu.reader.data.Clip
import io.github.bunyu.reader.data.Library
import io.github.bunyu.reader.data.OpenBook
import io.github.bunyu.reader.data.ReaderSettings
import io.github.bunyu.reader.data.Settings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class NarrationState(
    val bookId: String? = null,
    val title: String = "",
    val chapter: Int = 0,
    val chapterTitle: String = "",
    val piece: Int = 0,
    val totalPieces: Int = 0,
    /** The listener asked to play (it may still be waiting for audio). */
    val playing: Boolean = false,
    /** Playback has caught up with the voice engine and is waiting for the next piece. */
    val waiting: Boolean = false,
    /** Pieces ready from the current position onward, without a gap. */
    val readyAhead: Int = 0,
    val rendering: Boolean = false,
    val preparingAll: Boolean = false,
    /** Pieces ready between [preparedFrom] (where "prepare the book" started) and the end of the book. */
    val preparedTotal: Int = 0,
    val preparedFrom: Int = 0,
    val preparedFromChapter: Int = 0,
    val sleepAt: Long? = null,
    val sleepAtChapterEnd: Boolean = false,
    val skipped: Int = 0,
    /** Seconds of speech the phone generates per second of work; null until a passage was generated. */
    val speed: Float? = null,
    /** Set while playback holds back after running dry: how many passages are ready again, out of [Narration.HOLD_PIECES]. */
    val buffered: Int? = null,
    val error: String? = null,
    val finished: Boolean = false,
)

/**
 * Owns the narration: which piece is playing, which pieces are being generated ahead of it, the
 * background music and the sleep timer. Lives for the process; [ReaderService] keeps the process
 * in the foreground while it plays or prepares.
 */
class Narration(private val context: Context, private val library: Library, private val settings: ReaderSettings, private val renderer: Renderer) {
    private val main = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val mutable = MutableStateFlow(NarrationState())
    val state: StateFlow<NarrationState> = mutable.asStateFlow()

    val player: ExoPlayer = ExoPlayer.Builder(context)
        // Audio brought from a PC is played in stretches cut out of long files, so a jump must land on the exact time, in MP3 too.
        .setMediaSourceFactory(DefaultMediaSourceFactory(context, DefaultExtractorsFactory().setMp3ExtractorFlags(Mp3Extractor.FLAG_ENABLE_INDEX_SEEKING)))
        .setAudioAttributes(AudioAttributes.Builder().setUsage(C.USAGE_MEDIA).setContentType(C.AUDIO_CONTENT_TYPE_SPEECH).build(), true)
        .setHandleAudioBecomingNoisy(true)
        .setWakeMode(C.WAKE_MODE_LOCAL)
        .build()
    // Music never takes audio focus: it follows the narration instead of competing with it.
    private val music: ExoPlayer = ExoPlayer.Builder(context)
        .setAudioAttributes(AudioAttributes.Builder().setUsage(C.USAGE_MEDIA).setContentType(C.AUDIO_CONTENT_TYPE_MUSIC).build(), false)
        .build().apply { repeatMode = Player.REPEAT_MODE_ALL }

    private var book: OpenBook? = null
    private var voice: ResolvedVoice? = null
    private var queuedUntil = -1          // last book piece index added to the player
    /** Set when "prepare the book" starts, so the render loop counts what is already ready from that point. */
    @Volatile private var recount = false
    private var position = 0              // book piece index being heard
    private val wake = Channel<Unit>(Channel.CONFLATED)
    private var renderJob: Job? = null
    private var sleepChapter = -1
    /** Parts of a piece the listener was waiting for, played before the whole piece was ready: book piece index and file. */
    private val earlyDir = File(context.cacheDir, "early-parts")
    private val earlyFiles = ArrayDeque<Pair<Int, File>>()
    /**
     * Playback ran dry on a phone that generates slower than it plays: the player item to resume from once a
     * few passages are ready again, so the listener waits once instead of after every passage. -1 when not holding.
     */
    private var holdFrom = -1

    init {
        player.addListener(object : Player.Listener {
            override fun onMediaItemTransition(mediaItem: MediaItem?, reason: Int) {
                val index = mediaItem?.mediaId?.toIntOrNull() ?: return
                arrived(index)
            }
            override fun onPlaybackStateChanged(playbackState: Int) {
                if (playbackState == Player.STATE_ENDED) caughtUp()
                if (playbackState == Player.STATE_READY) mutable.value = mutable.value.copy(waiting = false)
            }
            override fun onPlayWhenReadyChanged(playWhenReady: Boolean, reason: Int) {
                mutable.value = mutable.value.copy(playing = playWhenReady)
                syncMusic()
                if (playWhenReady) ReaderService.start(context)
            }
            override fun onPlayerError(error: PlaybackException) {
                val file = player.currentMediaItem?.localConfiguration?.uri?.path?.let(::File)
                mutable.value = mutable.value.copy(error = "playback")
                // Audio the reader brought in is never deleted, and playing it again would only fail again.
                if (file != null && book?.pieces?.any { it.clip?.file == file } == true) { player.playWhenReady = false; return }
                // A damaged cached piece: delete it so it is generated again, and carry on.
                file?.delete()
                reload(position, play = mutable.value.playing)
            }
        })
        scope.launch {
            settings.flow.map { listOf(it.voice, it.dialect, it.pronunciations, it.fast) }.distinctUntilChanged().collect {
                if (book != null) { voice = null; reload(position, play = player.playWhenReady) }
            }
        }
        // Changed pauses change the audio too. The dialog moves in 0.1 s taps, so wait until the taps stop.
        scope.launch {
            settings.flow.map { it.pauses }.distinctUntilChanged().collectLatest {
                delay(PAUSE_SETTLE_MS)
                if (book != null) { voice = null; reload(position, play = player.playWhenReady) }
            }
        }
        scope.launch { settings.flow.collect { applyAudio(it) } }
        // With listening hidden the book is only read: nothing plays and no audio is prepared (unless the whole book was asked for).
        scope.launch {
            settings.flow.map { it.listening to it.readyWhileReading }.distinctUntilChanged().collect { (on, _) ->
                if (book == null) return@collect
                if (!on) player.playWhenReady = false
                reload(position, play = false)
            }
        }
        scope.launch { renderer.speed.collect { mutable.value = mutable.value.copy(speed = it) } }
    }

    private fun applyAudio(s: Settings) {
        player.setPlaybackSpeed(s.speed)
        music.volume = s.musicVolume
        val uri = s.musicUri?.takeIf { s.musicOn }
        val current = music.currentMediaItem?.localConfiguration?.uri?.toString()
        if (uri == null) { music.stop(); music.clearMediaItems() }
        else if (uri != current) { music.setMediaItem(MediaItem.fromUri(Uri.parse(uri))); music.prepare() }
        syncMusic()
    }

    private fun syncMusic() {
        music.playWhenReady = player.playWhenReady && settings.value.musicOn && settings.value.musicUri != null
    }

    /** Opens a book at its saved position (or [chapter] when given) without starting playback. */
    suspend fun open(bookId: String, chapter: Int? = null) {
        if (book?.book?.id == bookId && chapter == null) return
        val opened = library.open(bookId)
        val summary = library.books.value.firstOrNull { it.id == bookId }
        book = opened
        val start = when {
            chapter != null -> opened.chapterStarts.getOrElse(chapter) { 0 }
            summary != null -> summary.piece.coerceIn(0, (opened.pieces.size - 1).coerceAtLeast(0))
            else -> 0
        }
        mutable.value = NarrationState(bookId = bookId, title = opened.book.title, totalPieces = opened.pieces.size, speed = renderer.speed.value)
        reload(start, play = false)
    }

    fun current(): OpenBook? = book

    fun play() {
        if (book == null) return
        mutable.value = mutable.value.copy(error = null, finished = false)
        // Nothing to play at this place yet: start over so the piece is made the quick way, part by part,
        // even if it was already being made whole for the cache.
        if (renderJob?.isActive != true || (queuedUntil < position && !player.playWhenReady)) reload(position, play = true)
        player.playWhenReady = true
        if (player.playbackState == Player.STATE_IDLE) player.prepare()
        if (mutable.value.waiting) wake.trySend(Unit)
        ReaderService.start(context)
    }
    fun pause() { player.playWhenReady = false }
    fun toggle() = if (player.playWhenReady) pause() else play()

    fun seekPiece(index: Int) { reload(index, play = player.playWhenReady) }
    fun seekChapter(chapter: Int) { book?.let { reload(it.chapterStarts.getOrElse(chapter) { 0 }, play = player.playWhenReady) } }
    /** The reader scrolled to [index] while only reading: remember the place, so listening would start there. */
    fun readAt(index: Int) {
        val opened = book ?: return
        if (settings.value.listening || player.playWhenReady || index == position || index !in opened.pieces.indices) return
        reload(index, play = false)
        scope.launch { library.savePosition(opened.book.id, opened.pieces[index].chapter, index) }
    }
    fun next() = seekPiece((position + 1).coerceAtMost((book?.pieces?.size ?: 1) - 1))
    fun previous() = seekPiece((position - 1).coerceAtLeast(0))

    /** Generate every remaining piece of the book, e.g. overnight while charging, so playback never waits. */
    fun prepareAll(on: Boolean) {
        // From where the listener is to the end: earlier chapters are already heard (or skipped on purpose).
        val from = position
        val chapter = book?.pieces?.getOrNull(from)?.chapter ?: 0
        mutable.value = mutable.value.copy(preparingAll = on, preparedFrom = from, preparedFromChapter = chapter,
            preparedTotal = 0)
        recount = on
        if (on) {
            ReaderService.start(context)
            if (renderJob?.isActive == true) wake.trySend(Unit) else reload(position, play = player.playWhenReady)
        }
    }

    fun sleepAfter(minutes: Int?) {
        main.removeCallbacks(sleepFade)
        sleepFade.step = 0
        player.volume = 1f; music.volume = settings.value.musicVolume
        sleepChapter = -1
        if (minutes == null) { mutable.value = mutable.value.copy(sleepAt = null, sleepAtChapterEnd = false); return }
        val at = System.currentTimeMillis() + minutes * 60_000L
        mutable.value = mutable.value.copy(sleepAt = at, sleepAtChapterEnd = false)
        main.postDelayed(sleepFade, minutes * 60_000L)
    }
    fun sleepAtChapterEnd() {
        main.removeCallbacks(sleepFade)
        sleepChapter = book?.pieces?.getOrNull(position)?.chapter ?: -1
        mutable.value = mutable.value.copy(sleepAt = null, sleepAtChapterEnd = true)
    }

    private val sleepFade = object : Runnable {
        var step = 0
        override fun run() {
            // Fade both narration and music over about ten seconds, then stop and restore levels.
            step++
            val level = 1f - step / 20f
            if (level <= 0f) {
                step = 0
                player.playWhenReady = false
                player.volume = 1f; music.volume = settings.value.musicVolume
                mutable.value = mutable.value.copy(sleepAt = null)
                return
            }
            player.volume = level; music.volume = settings.value.musicVolume * level
            main.postDelayed(this, 500)
        }
    }

    private fun arrived(index: Int) {
        val opened = book ?: return
        val piece = opened.pieces.getOrNull(index) ?: return
        if (mutable.value.sleepAtChapterEnd && sleepChapter >= 0 && piece.chapter != sleepChapter) {
            player.playWhenReady = false
            sleepChapter = -1
            mutable.value = mutable.value.copy(sleepAtChapterEnd = false)
        }
        position = index
        // Drop pieces already heard so the player's list stays short on long books.
        val currentItem = player.currentMediaItemIndex
        if (currentItem > 3) player.removeMediaItems(0, currentItem - 3)
        // The three items kept behind the current one belong to the last three pieces at most.
        while (earlyFiles.isNotEmpty() && earlyFiles.first().first < index - 4) earlyFiles.removeFirst().second.delete()
        publish()
        scope.launch { library.savePosition(opened.book.id, piece.chapter, index) }
        wake.trySend(Unit)
    }

    private fun caughtUp() {
        val opened = book ?: return
        if (queuedUntil >= opened.pieces.size - 1) {
            mutable.value = mutable.value.copy(finished = true, playing = false)
            player.playWhenReady = false
            return
        }
        mutable.value = mutable.value.copy(waiting = true)
        if (player.playWhenReady && holdFrom < 0 && (renderer.speed.value ?: 1f) < 1f) {
            holdFrom = player.mediaItemCount
            mutable.value = mutable.value.copy(buffered = 0)
        }
        wake.trySend(Unit)
    }

    private fun publish() {
        val opened = book ?: return
        val piece = opened.pieces.getOrNull(position)
        mutable.value = mutable.value.copy(
            piece = position, chapter = piece?.chapter ?: 0,
            chapterTitle = piece?.let { opened.book.chapters[it.chapter].title }.orEmpty(),
            readyAhead = (queuedUntil - position + 1).coerceAtLeast(0),
            waiting = mutable.value.waiting || (player.playWhenReady && queuedUntil < position),
        )
    }

    /** Restarts the player at [index]: queues whatever is already generated and lets the render loop do the rest. */
    private fun reload(index: Int, play: Boolean) {
        val opened = book ?: return
        position = index.coerceIn(0, (opened.pieces.size - 1).coerceAtLeast(0))
        queuedUntil = position - 1
        player.stop(); player.clearMediaItems()
        earlyFiles.clear()
        holdFrom = -1
        mutable.value = mutable.value.copy(buffered = null)
        earlyDir.listFiles()?.forEach { it.delete() }
        player.playWhenReady = play
        renderJob?.cancel()
        renderJob = if (play || settings.value.listening || settings.value.readyWhileReading || mutable.value.preparingAll) scope.launch { renderLoop() } else null
        publish()
    }

    private suspend fun ensureVoice(): ResolvedVoice? {
        voice?.let { return it }
        return try {
            renderer.resolve(settings.value).also { voice = it }
        } catch (e: VoiceUnavailable) {
            mutable.value = mutable.value.copy(error = e.reason, waiting = false); player.playWhenReady = false; null
        }
    }

    /** Queues every contiguous generated piece after [queuedUntil]. */
    private fun queueReady(voice: ResolvedVoice?) {
        val opened = book ?: return
        val folder = library.audioDir(opened.book.id)
        val startCount = player.mediaItemCount
        while (queuedUntil + 1 < opened.pieces.size && queuedUntil + 1 - position < 400) {
            val piece = opened.pieces[queuedUntil + 1]
            val clip = piece.clip
            val file = if (clip != null) clip.file else voice?.let { renderer.file(folder, it, piece) }?.takeIf { it.isFile } ?: break
            player.addMediaItem(item(opened, piece.index, file, clip))
            queuedUntil++
        }
        if (player.mediaItemCount > startCount) resume(startCount)
        publish()
    }

    private fun item(opened: OpenBook, index: Int, file: File, clip: Clip? = null): MediaItem {
        val meta = MediaMetadata.Builder().setTitle(opened.book.chapters[opened.pieces[index].chapter].title).setArtist(opened.book.title).build()
        val item = MediaItem.Builder().setMediaId(index.toString()).setUri(Uri.fromFile(file)).setMediaMetadata(meta)
        if (clip != null) item.setClippingConfiguration(MediaItem.ClippingConfiguration.Builder().setStartPositionMs(clip.startMs).setEndPositionMs(maxOf(clip.endMs, clip.startMs + 1)).build())
        return item.build()
    }

    /** Gets the player going again after items were added from [firstNew] on. */
    private fun resume(firstNew: Int) {
        if (player.playbackState == Player.STATE_IDLE) player.prepare()
        if (player.playbackState != Player.STATE_ENDED) return
        if (holdFrom < 0) { player.seekTo(firstNew, 0); return }
        val ready = queuedUntil - position
        if (ready < HOLD_PIECES && queuedUntil < (book?.pieces?.size ?: 0) - 1) { mutable.value = mutable.value.copy(buffered = ready.coerceAtLeast(0)); return }
        player.seekTo(holdFrom, 0)
        holdFrom = -1
        mutable.value = mutable.value.copy(buffered = null)
    }

    /** The book is only being read: the player is hidden and nothing plays. */
    private fun quiet() = !settings.value.listening && !player.playWhenReady

    private suspend fun renderLoop() {
        val opened = book ?: return
        val folder = library.audioDir(opened.book.id)
        // Only reading: wait until the reader has settled on a place before spending the battery on it.
        if (quiet() && !mutable.value.preparingAll) delay(READY_DELAY_MS)
        // A book that came with all its audio needs no voice; a piece without audio is read by the model as usual.
        val resolved = if (opened.pieces.all { it.clip != null }) null else ensureVoice() ?: return
        queueReady(resolved)
        if (mutable.value.preparingAll) mutable.value = mutable.value.copy(preparedTotal = countReady(opened, folder, resolved, mutable.value.preparedFrom))
        while (true) {
            if (recount && mutable.value.preparingAll) {
                recount = false
                mutable.value = mutable.value.copy(preparedTotal = countReady(opened, folder, resolved, mutable.value.preparedFrom))
            }
            val ahead = if (mutable.value.preparingAll) opened.pieces.size else if (quiet()) READY_WHILE_READING - 1 else settings.value.prepareAhead
            val limit = (position + ahead).coerceAtMost(opened.pieces.size - 1)
            val target = (queuedUntil + 1..limit).firstOrNull { resolved != null && opened.pieces[it].clip == null && !renderer.file(folder, resolved, opened.pieces[it]).isFile }
            if (target == null || resolved == null) {
                mutable.value = mutable.value.copy(rendering = false)
                if (mutable.value.preparingAll && limit >= opened.pieces.size - 1) mutable.value = mutable.value.copy(preparingAll = false)
                wake.receive(); queueReady(resolved); continue
            }
            mutable.value = mutable.value.copy(rendering = true)
            var played = 0
            // Little is ready ahead of the listener: hand this piece to the player part by part as it is generated,
            // so a long piece after short ones starts on time instead of only when all of it is done. When the
            // listener is on the last ready piece, or already waiting, the first words are also made the quick way.
            val next = target == queuedUntil + 1 && holdFrom < 0 && player.playWhenReady
            val early = if (next && queuedUntil - position < STREAM_AHEAD) EarlyParts(earlyDir, hurry = queuedUntil <= position) { part ->
                withContext(Dispatchers.Main) {
                    val count = player.mediaItemCount
                    player.addMediaItem(item(opened, target, part))
                    earlyFiles.addLast(target to part)
                    played++
                    resume(count)
                }
            } else null
            try {
                renderer.render(folder, resolved, opened.pieces[target], early)
            } catch (e: CancellationException) { throw e }
            catch (e: VoiceUnavailable) {
                mutable.value = mutable.value.copy(error = e.reason, rendering = false); player.playWhenReady = false; return
            } catch (e: Exception) {
                withContext(Dispatchers.IO) { skip(folder, resolved, opened, target) }
                mutable.value = mutable.value.copy(skipped = mutable.value.skipped + 1)
            }
            if (mutable.value.preparingAll && target >= mutable.value.preparedFrom) mutable.value = mutable.value.copy(preparedTotal = mutable.value.preparedTotal + 1)
            if (played > 0) {
                // Its parts are in the player already, so the whole piece must not be queued a second time.
                queuedUntil = target
                if (player.playbackState == Player.STATE_ENDED) caughtUp()
            }
            queueReady(resolved)
        }
    }

    private suspend fun skip(folder: File, voice: ResolvedVoice, opened: OpenBook, index: Int) {
        val silent = opened.pieces[index].copy(text = "")
        val file = renderer.render(folder, voice, silent)
        file.copyTo(renderer.file(folder, voice, opened.pieces[index]), overwrite = true)
    }

    private suspend fun countReady(opened: OpenBook, folder: File, voice: ResolvedVoice?, from: Int) = withContext(Dispatchers.IO) {
        opened.pieces.drop(from).count { it.clip != null || (voice != null && renderer.file(folder, voice, it).isFile) }
    }

    companion object {
        /** Passages to have ready before playback resumes after running dry. */
        const val HOLD_PIECES = 3
        /** Passages kept ready at the reading place when the reader asked for that, and how long the place must stay put first. */
        const val READY_WHILE_READING = 2
        const val READY_DELAY_MS = 2000L
        /** With fewer passages than this ready after the one being heard, the next one is played part by part as it is made. */
        const val STREAM_AHEAD = 4
        const val PAUSE_SETTLE_MS = 700L
    }

    /** Forgets the open book, e.g. when it is deleted. */
    fun close() {
        stop()
        renderJob?.cancel()
        player.stop(); player.clearMediaItems()
        book = null; voice = null
        mutable.value = NarrationState()
    }

    /** Stops everything when the user closes the notification. */
    fun stop() {
        pause()
        prepareAll(false)
        sleepAfter(null)
    }
}
