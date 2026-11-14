package io.github.bunyu.reader.ui

import android.media.MediaPlayer
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import io.github.bunyu.reader.R
import io.github.bunyu.reader.ReaderApp
import io.github.bunyu.reader.data.VoiceChoice
import io.github.bunyu.reader.vieneu.DownloadFailed
import io.github.bunyu.reader.vieneu.EnrolmentFailed
import io.github.bunyu.reader.vieneu.NanoPreset
import io.github.bunyu.reader.vieneu.ReleaseFile
import io.github.bunyu.reader.voice.VoiceBundle
import io.github.bunyu.reader.voice.VoiceBundleException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VoicesScreen(modifier: Modifier, incoming: Uri?, consumed: () -> Unit, onSetup: () -> Unit) {
    val app = ReaderApp.get(LocalContext.current)
    val settings by app.settings.flow.collectAsState()
    val download by app.vieneu.assets.state.collectAsState()
    // Read again whenever a download starts or ends, so the list appears as soon as the model is installed.
    val presets by produceState<List<NanoPreset>?>(null, download.running) {
        value = if (app.vieneu.ready) runCatching { app.vieneu.presets() }.getOrNull()?.let { all ->
            val first = all.default
            listOfNotNull(first) + all.voices.filter { it !== first }
        } else null
    }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val cloned by app.voices.voices.collectAsState()
    /** A recording copied into the cache and waiting for a name: the clip and a suggested name. */
    var pending by remember { mutableStateOf<Pair<File, String>?>(null) }
    var problem by remember { mutableStateOf<Int?>(null) }
    var busy by remember { mutableStateOf(false) }
    var recording by remember { mutableStateOf(false) }
    var renaming by remember { mutableStateOf<io.github.bunyu.reader.voice.ReaderVoice?>(null) }
    /** The preset whose sample is being generated. */
    var sampling by remember { mutableStateOf<String?>(null) }
    val player = remember { mutableStateOf<MediaPlayer?>(null) }
    DisposableEffect(Unit) { onDispose { player.value?.release() } }
    fun play(file: File?) {
        player.value?.release(); player.value = null
        if (file != null) player.value = runCatching { MediaPlayer().apply { setDataSource(file.path); prepare(); start() } }.getOrNull()
    }
    fun sample(preset: NanoPreset) {
        if (sampling != null) return
        play(null); problem = null
        sampling = preset.name
        scope.launch {
            val file = runCatching { app.vieneu.sample(preset) }.getOrNull()
            sampling = null
            if (file != null) play(file) else problem = R.string.voice_sample_failed
        }
    }
    fun discard(clip: File) { clip.parentFile?.takeIf { it.name.startsWith("voice-clone-") }?.deleteRecursively() }

    /** Copies the picked file into the cache; a VoiceStudio export is opened and its reference clip taken. */
    fun receive(uri: Uri) {
        busy = true; problem = null
        scope.launch {
            val workspace = File(context.cacheDir, "voice-clone-" + UUID.randomUUID())
            try {
                pending = withContext(Dispatchers.IO) {
                    val name = displayName(context, uri)
                    val extension = name.substringAfterLast('.', "").lowercase().filter { it.isLetterOrDigit() }.take(8)
                    val copy = File(workspace, "clip." + extension.ifEmpty { "audio" }).apply { parentFile!!.mkdirs() }
                    (context.contentResolver.openInputStream(uri) ?: error("unreadable")).use { input ->
                        copy.outputStream().use { out ->
                            val buffer = ByteArray(262_144); var total = 0L
                            while (true) { val n = input.read(buffer); if (n < 0) break; total += n; if (total > 200L * 1024 * 1024) error("too large"); out.write(buffer, 0, n) }
                        }
                    }
                    if (isVoiceFile(name)) VoiceBundle.read(copy, File(workspace, "members")).let { it.audio to it.name }
                    else copy to name.substringBeforeLast('.').take(60)
                }
            } catch (e: VoiceBundleException) { workspace.deleteRecursively(); problem = R.string.voice_error_format }
            catch (e: Exception) { workspace.deleteRecursively(); problem = R.string.voice_error_audio }
            busy = false
        }
    }

    fun clone(clip: File, name: String) {
        busy = true; problem = null; pending = null
        // On the app's scope: enrolling must finish even if the user leaves this screen.
        app.scope.launch {
            try {
                if (!app.vieneu.canClone) app.vieneu.assets.download(ReleaseFile.CLONE, app.settings.value.allowMobileData)
                val voice = app.voices.add(clip, name) { app.vieneu.enroll(it) }
                app.settings.update { it.copy(voice = VoiceChoice.Cloned(voice.id)) }
            } catch (e: EnrolmentFailed) { problem = if (e.reason == "silent") R.string.voice_error_silent else R.string.voice_error_short }
            catch (e: DownloadFailed) { problem = if (e.reason == "wifi") R.string.engine_wifi else if (e.reason == "storage") R.string.engine_storage else R.string.voice_error_download }
            catch (e: kotlinx.coroutines.CancellationException) { throw e }
            catch (e: Exception) { problem = R.string.voice_error_audio }
            finally { withContext(Dispatchers.IO) { discard(clip) }; busy = false }
        }
    }
    LaunchedEffect(incoming) { incoming?.let { consumed(); receive(it) } }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(::receive) }
    val microphone = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) recording = true else problem = R.string.record_permission
    }
    fun record() {
        problem = null
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) recording = true
        else microphone.launch(Manifest.permission.RECORD_AUDIO)
    }

    Scaffold(modifier = modifier, topBar = { TopAppBar(title = { Text(stringResource(R.string.tab_voices)) }) }) { padding ->
        LazyColumn(Modifier.padding(padding), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val list = presets
            if (list == null) item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text(stringResource(R.string.voices_need_engine), style = MaterialTheme.typography.bodyMedium)
                        TextButton(onSetup) { Text(stringResource(R.string.tab_settings)) }
                    }
                }
            } else {
                item {
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp)) {
                            Text(stringResource(R.string.voices_your), style = MaterialTheme.typography.titleMedium)
                            Text(stringResource(R.string.voices_clone_how), style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(vertical = 6.dp))
                            if (!app.vieneu.canClone) Text(stringResource(R.string.voice_clone_download, app.vieneu.release.bytes(ReleaseFile.CLONE) / 1_000_000), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 6.dp)) {
                                Button(onClick = ::record, enabled = !busy) {
                                    Icon(painterResource(R.drawable.ic_voice), null); Text(stringResource(R.string.record_voice))
                                }
                                OutlinedButton(onClick = { pick.launch(arrayOf("audio/*", "application/octet-stream", "application/zip")) }, enabled = !busy) {
                                    Icon(Icons.Filled.Add, null); Text(stringResource(R.string.voice_from_file))
                                }
                            }
                            if (busy) {
                                if (download.running && download.total > 0) LinearProgressIndicator(progress = { download.received.toFloat() / download.total }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                                else LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp))
                                Text(stringResource(if (download.running) R.string.engine_downloading else R.string.voice_working, download.received / 1_000_000, download.total / 1_000_000), style = MaterialTheme.typography.labelSmall)
                            }
                            problem?.let { Text(stringResource(it), color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp)) }
                        }
                    }
                }
                items(cloned, key = { "clone:" + it.id }) { voice ->
                    val choice = VoiceChoice.Cloned(voice.id)
                    Row(Modifier.fillMaxWidth().clickable { app.settings.update { it.copy(voice = choice) } }, verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(settings.voice == choice, { app.settings.update { it.copy(voice = choice) } })
                        Text(voice.name, Modifier.weight(1f))
                        IconButton({ renaming = voice }) { Icon(Icons.Filled.Edit, stringResource(R.string.voice_rename)) }
                        IconButton({ play(app.voices.preview(voice)) }) { Icon(painterResource(R.drawable.ic_play), stringResource(R.string.voice_preview)) }
                        IconButton({ scope.launch { app.voices.delete(voice); if (app.settings.value.voice == choice) app.settings.update { it.copy(voice = null) } } }) { Icon(Icons.Filled.Delete, stringResource(R.string.delete)) }
                    }
                }
                item {
                    Text(stringResource(R.string.voices_presets), style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 8.dp))
                    Text(stringResource(R.string.voices_presets_hint), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                items(list, key = { it.name }) { preset ->
                    val choice = VoiceChoice.Preset(preset.name)
                    val selected = settings.voice == choice || (cloned.none { VoiceChoice.Cloned(it.id) == settings.voice } && list.none { VoiceChoice.Preset(it.name) == settings.voice } && preset === list.first())
                    Row(Modifier.fillMaxWidth().clickable { app.settings.update { it.copy(voice = choice) } }, verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selected, { app.settings.update { it.copy(voice = choice) } })
                        Column(Modifier.weight(1f)) {
                            Text(preset.name)
                            if (preset.description.isNotBlank()) Text(preset.description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        if (sampling == preset.name) CircularProgressIndicator(Modifier.padding(12.dp).size(24.dp), strokeWidth = 2.dp)
                        else IconButton({ sample(preset) }, enabled = sampling == null) { Icon(painterResource(R.drawable.ic_play), stringResource(R.string.voice_sample)) }
                    }
                }
            }
        }
    }
    if (recording) RecordDialog(onDismiss = { recording = false }) { clip ->
        recording = false
        pending = clip to context.getString(R.string.record_default_name)
    }
    renaming?.let { voice ->
        RenameDialog(voice.name, onDismiss = { renaming = null }) { name ->
            renaming = null
            scope.launch { app.voices.rename(voice, name) }
        }
    }
    pending?.let { (clip, suggested) ->
        CloneDialog(suggested, onDismiss = { pending = null; scope.launch(Dispatchers.IO) { discard(clip) } }) { name -> clone(clip, name) }
    }
}

@Composable
private fun CloneDialog(suggested: String, onDismiss: () -> Unit, onConfirm: (String) -> Unit) {
    var name by remember { mutableStateOf(suggested) }
    var consent by remember { mutableStateOf(false) }
    AlertDialog(onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.voice_clone)) },
        text = {
            Column {
                OutlinedTextField(name, { name = it }, label = { Text(stringResource(R.string.voice_name)) }, singleLine = true, modifier = Modifier.fillMaxWidth())
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.clickable { consent = !consent }.padding(top = 8.dp)) {
                    Checkbox(consent, { consent = it })
                    Text(stringResource(R.string.voice_consent), style = MaterialTheme.typography.bodyMedium)
                }
            }
        },
        confirmButton = { TextButton({ onConfirm(name) }, enabled = consent && name.isNotBlank()) { Text(stringResource(R.string.voice_clone)) } },
        dismissButton = { TextButton(onDismiss) { Text(stringResource(R.string.cancel)) } })
}

@Composable
private fun RenameDialog(current: String, onDismiss: () -> Unit, onConfirm: (String) -> Unit) {
    var name by remember { mutableStateOf(current) }
    AlertDialog(onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.voice_rename)) },
        text = { OutlinedTextField(name, { name = it }, label = { Text(stringResource(R.string.voice_name)) }, singleLine = true, modifier = Modifier.fillMaxWidth()) },
        confirmButton = { TextButton({ onConfirm(name) }, enabled = name.isNotBlank() && name.trim() != current) { Text(stringResource(R.string.save)) } },
        dismissButton = { TextButton(onDismiss) { Text(stringResource(R.string.cancel)) } })
}
