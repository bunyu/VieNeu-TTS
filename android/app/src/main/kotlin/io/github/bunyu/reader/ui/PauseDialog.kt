package io.github.bunyu.reader.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import io.github.bunyu.reader.R
import io.github.bunyu.reader.text.PauseRules
import java.util.Locale

/** "Thiết lập ngắt nghỉ": seconds of silence per punctuation mark, 0.1 s steps. */
@Composable
fun PauseDialog(current: PauseRules, onChange: (PauseRules) -> Unit, onDismiss: () -> Unit) {
    var rules by remember { mutableStateOf(current) }
    fun set(next: PauseRules) { rules = next; onChange(next) }
    val labels = mapOf(
        "." to R.string.pause_period, "," to R.string.pause_comma, "?" to R.string.pause_question, "!" to R.string.pause_exclamation,
        ":" to R.string.pause_colon, ";" to R.string.pause_semicolon, PauseRules.ELLIPSIS to R.string.pause_ellipsis, PauseRules.NEW_LINE to R.string.pause_new_line,
    )
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.pause_title)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text(stringResource(R.string.pause_subtitle), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                for ((mark, label) in labels) {
                    val value = rules.seconds(mark)
                    val changed = mark in rules.custom
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 6.dp)) {
                        Text(stringResource(label) + "  " + if (mark == PauseRules.NEW_LINE) "↵" else mark, Modifier.weight(1f),
                            fontWeight = if (changed) FontWeight.SemiBold else FontWeight.Normal)
                        FilledTonalIconButton({ set(rules.with(mark, value - 0.1)) }, enabled = value > 0.0) { Text("−") }
                        Text(String.format(Locale.ROOT, "%.1f", value).removeSuffix(".0") + " s", Modifier.width(52.dp), textAlign = TextAlign.Center,
                            color = if (changed) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface)
                        FilledTonalIconButton({ set(rules.with(mark, value + 0.1)) }, enabled = value < PauseRules.MAX_SECONDS) { Text("+") }
                    }
                }
                Text(stringResource(R.string.pause_tags), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 10.dp))
            }
        },
        confirmButton = { TextButton(onDismiss) { Text(stringResource(R.string.pause_done)) } },
        dismissButton = { TextButton({ set(PauseRules()) }, enabled = rules.custom.isNotEmpty()) { Text(stringResource(R.string.pause_reset)) } },
    )
}
