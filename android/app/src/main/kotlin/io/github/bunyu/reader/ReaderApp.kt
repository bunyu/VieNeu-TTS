package io.github.bunyu.reader

import android.app.Application
import android.content.Context
import io.github.bunyu.reader.data.Library
import io.github.bunyu.reader.data.ReaderSettings
import io.github.bunyu.reader.playback.VieNeu
import io.github.bunyu.reader.playback.Narration
import io.github.bunyu.reader.playback.Renderer
import io.github.bunyu.reader.voice.ReaderVoices
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File

class ReaderApp : Application() {
    /** Work that must outlive a screen, such as model downloads. */
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    val settings by lazy { ReaderSettings(this) }
    val library by lazy { Library(this) }
    val voices by lazy { ReaderVoices(File(filesDir, "voices")) }
    /** The user's recordings kept with their text, for fine-tuning a voice on the PC. */
    val training by lazy { io.github.bunyu.reader.voice.TrainingTakes(File(filesDir, "training")) }
    val vieneu by lazy { VieNeu(this) }
    val narration by lazy { Narration(this, library, settings, Renderer(vieneu, voices)) }

    override fun onCreate() {
        super.onCreate()
        scope.launch { library.refresh(); voices.refresh(); training.refresh() }
    }

    companion object {
        fun get(context: Context) = context.applicationContext as ReaderApp
    }
}
