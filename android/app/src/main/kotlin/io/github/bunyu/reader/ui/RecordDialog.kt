package io.github.bunyu.reader.ui

import android.media.MediaPlayer
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.foundation.layout.Row
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import io.github.bunyu.reader.R
import io.github.bunyu.reader.media.VoiceRecorder
import java.io.File
import java.util.UUID

/**
 * Records the user reading a Vietnamese passage aloud, then hands the clip on to be named and enrolled.
 * The passage uses all six tones in natural sentences and takes about 25 seconds to read.
 */
@Composable
fun RecordDialog(onDismiss: () -> Unit, onRecorded: (File) -> Unit) {
    val context = LocalContext.current
    val recorder = remember { VoiceRecorder() }
    val level by recorder.level.collectAsState()
    val seconds by recorder.seconds.collectAsState()
    var active by remember { mutableStateOf(false) }
    var clip by remember { mutableStateOf<File?>(null) }
    var failed by remember { mutableStateOf(false) }
    val player = remember { mutableStateOf<MediaPlayer?>(null) }
    val workspace = remember { File(context.cacheDir, "voice-clone-" + UUID.randomUUID()) }
    fun stopPlayback() { player.value?.release(); player.value = null }
    DisposableEffect(Unit) {
        onDispose { recorder.stop(); stopPlayback() }
    }
    fun start() {
        stopPlayback(); failed = false; clip = null
        failed = runCatching { recorder.start() }.isFailure
        active = !failed
    }
    fun finish() {
        recorder.stop(); active = false
        clip = recorder.save(File(workspace, "recording.wav"))
    }
    // The recorder stops by itself at its limit.
    LaunchedEffect(seconds) { if (active && seconds >= VoiceRecorder.MAX_SECONDS) finish() }

    AlertDialog(
        onDismissRequest = {},
        title = { Text(stringResource(R.string.record_title)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text(stringResource(R.string.record_tips), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Card(Modifier.fillMaxWidth().padding(vertical = 8.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                    Text(PASSAGE, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(12.dp).heightIn(min = 120.dp))
                }
                LinearProgressIndicator(progress = { if (active) level.coerceIn(0f, 1f) else 0f }, modifier = Modifier.fillMaxWidth())
                val status = when {
                    failed -> stringResource(R.string.record_failed)
                    active -> stringResource(R.string.record_recording, seconds)
                    clip != null && seconds < 8 -> stringResource(R.string.record_too_short, seconds)
                    clip != null -> stringResource(R.string.record_done, seconds)
                    else -> stringResource(R.string.record_ready)
                }
                Text(status, style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 4.dp),
                    color = if (failed || (clip != null && seconds < 8)) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                Row(Modifier.padding(top = 4.dp)) {
                    when {
                        active -> TextButton(::finish) { Text(stringResource(R.string.record_stop)) }
                        clip != null -> {
                            TextButton({
                                stopPlayback()
                                player.value = runCatching { MediaPlayer().apply { setDataSource(clip!!.path); prepare(); start() } }.getOrNull()
                            }) { Text(stringResource(R.string.record_play)) }
                            TextButton(::start) { Text(stringResource(R.string.record_again)) }
                        }
                        else -> TextButton(::start) { Text(stringResource(R.string.record_start)) }
                    }
                }
            }
        },
        confirmButton = {
            TextButton({ stopPlayback(); clip?.let(onRecorded) }, enabled = !active && clip != null && seconds >= 2) { Text(stringResource(R.string.record_use)) }
        },
        dismissButton = {
            TextButton({ recorder.stop(); stopPlayback(); workspace.deleteRecursively(); onDismiss() }) { Text(stringResource(R.string.cancel)) }
        },
    )
}

private const val PASSAGE =
    "Hôm nay trời đẹp, tôi ngồi bên cửa sổ đọc sách. Ngoài vườn, mấy chú chim sẻ nhảy nhót trên cành bưởi, " +
    "còn bà ngoại đang tưới rau. Tôi nhớ những ngày thơ ấu ở quê, con đường đất đỏ dẫn ra bến sông và tiếng " +
    "gọi đò vọng lại mỗi chiều. Dù đã đi xa, những kỷ niệm ấy vẫn còn nguyên vẹn trong lòng tôi. Mỗi khi " +
    "mệt mỏi, tôi chỉ cần nhắm mắt lại là thấy mình đang đứng giữa cánh đồng lúa chín vàng."
