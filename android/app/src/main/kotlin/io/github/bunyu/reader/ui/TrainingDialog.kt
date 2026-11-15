package io.github.bunyu.reader.ui

import android.media.MediaPlayer
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import io.github.bunyu.reader.R
import io.github.bunyu.reader.ReaderApp
import io.github.bunyu.reader.media.VoiceRecorder
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID

/** The sentences offered for fine-tuning recordings: the same list as the Windows app's fine-tune tab. */
fun trainingSentences(context: android.content.Context): List<String> =
    runCatching { context.assets.open("finetune_sentences.txt").bufferedReader().readLines().map { it.trim() }.filter { it.isNotEmpty() } }.getOrDefault(emptyList())

/**
 * Records sentences one by one for fine-tuning a voice on the PC. Each take is kept with its text under the voice
 * [initialName]; the set is exported from the Voices screen as a `.dtdata` file for the Windows app.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun TrainingDialog(initialName: String, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val app = ReaderApp.get(context)
    val scope = rememberCoroutineScope()
    val sets by app.training.sets.collectAsState()
    val recorder = remember { VoiceRecorder() }
    val level by recorder.level.collectAsState()
    val seconds by recorder.seconds.collectAsState()
    val sentences = remember { trainingSentences(context) }
    var name by remember { mutableStateOf(initialName) }
    var at by remember { mutableIntStateOf(0) }
    var active by remember { mutableStateOf(false) }
    var take by remember { mutableStateOf<File?>(null) }
    var message by remember { mutableStateOf<Int?>(null) }
    val player = remember { mutableStateOf<MediaPlayer?>(null) }
    val workspace = remember { File(context.cacheDir, "training-" + UUID.randomUUID()) }
    fun stopPlayback() { player.value?.release(); player.value = null }
    DisposableEffect(Unit) { onDispose { recorder.stop(); stopPlayback(); workspace.deleteRecursively() } }

    val current = sets.firstOrNull { it.name.trim().equals(name.trim(), ignoreCase = true) }
    val done = current?.takes?.map { it.text }?.toSet().orEmpty()
    // The next sentence not yet recorded for this voice, from where the reader is.
    LaunchedEffect(name, done.size) { while (at < sentences.size && sentences[at] in done) at++ }
    val sentence = sentences.getOrNull(at)

    fun start() {
        stopPlayback(); take = null; message = null
        active = runCatching { recorder.start() }.isSuccess
        if (!active) message = R.string.record_failed
    }
    fun finish() { recorder.stop(); active = false; take = recorder.save(File(workspace, "take.wav")) }
    LaunchedEffect(seconds) { if (active && seconds >= MAX_TAKE) finish() }

    AlertDialog(
        onDismissRequest = {},
        title = { Text(stringResource(R.string.training_title)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text(stringResource(R.string.training_tips), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                OutlinedTextField(name, { name = it; take = null }, label = { Text(stringResource(R.string.voice_name)) }, singleLine = true,
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp), enabled = !active)
                Text(stringResource(R.string.training_progress, current?.takes?.size ?: 0, (current?.seconds ?: 0.0) / 60),
                    style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 6.dp))
                Card(Modifier.fillMaxWidth().padding(vertical = 8.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                    Text(sentence ?: stringResource(R.string.training_all_done), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(12.dp).heightIn(min = 72.dp))
                }
                LinearProgressIndicator(progress = { if (active) level.coerceIn(0f, 1f) else 0f }, modifier = Modifier.fillMaxWidth())
                Text(when {
                    message != null -> stringResource(message!!)
                    active -> stringResource(R.string.record_recording, seconds)
                    take != null -> stringResource(R.string.training_take, seconds)
                    else -> stringResource(R.string.training_ready)
                }, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 4.dp),
                    color = if (message != null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.padding(top = 4.dp)) {
                    when {
                        active -> TextButton(::finish) { Text(stringResource(R.string.record_stop)) }
                        sentence == null -> {}
                        take != null -> {
                            TextButton({
                                stopPlayback()
                                // Not apply { start() }: there a bare start() is this dialog's own start(), which records.
                                player.value = runCatching { MediaPlayer().also { it.setDataSource(take!!.path); it.prepare(); it.start() } }.getOrNull()
                            }) { Text(stringResource(R.string.record_play)) }
                            TextButton(::start) { Text(stringResource(R.string.record_again)) }
                            TextButton({
                                val recorded = take ?: return@TextButton
                                // The PC trains on clips of 3 to 15 seconds.
                                if (seconds < MIN_TAKE) { message = R.string.training_too_short; return@TextButton }
                                stopPlayback()
                                scope.launch {
                                    app.training.add(name, sentence, recorded)
                                    take = null; message = null; at++
                                }
                            }, enabled = name.isNotBlank()) { Text(stringResource(R.string.training_save)) }
                        }
                        else -> {
                            TextButton(::start, enabled = name.isNotBlank()) { Text(stringResource(R.string.record_start)) }
                            TextButton({ take = null; message = null; at++ }) { Text(stringResource(R.string.training_skip)) }
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton({ recorder.stop(); stopPlayback(); onDismiss() }) { Text(stringResource(R.string.close)) } },
    )
}

private const val MIN_TAKE = 3.0
private const val MAX_TAKE = 15
