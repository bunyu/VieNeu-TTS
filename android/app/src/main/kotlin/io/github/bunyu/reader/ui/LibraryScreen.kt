package io.github.bunyu.reader.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.bunyu.reader.R
import io.github.bunyu.reader.ReaderApp
import io.github.bunyu.reader.data.BookSummary
import io.github.bunyu.reader.data.PreparedAudio
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedButton
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LibraryScreen(modifier: Modifier, importing: Boolean, onOpen: (String) -> Unit, onImport: (Uri) -> Unit) {
    val app = ReaderApp.get(LocalContext.current)
    val books by app.library.books.collectAsState()
    val narration by app.narration.state.collectAsState()
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(onImport) }
    val scope = rememberCoroutineScope()
    var linking by remember { mutableStateOf(false) }
    /** Files of a shared folder downloaded so far, and how many there are. */
    var fetched by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var fetching by remember { mutableStateOf(false) }
    var notice by remember { mutableStateOf<String?>(null) }
    val context = LocalContext.current
    /** Audio made on a PC: runs the import and says what came of it. */
    fun prepared(import: suspend () -> PreparedAudio.Result) {
        fetching = true
        scope.launch {
            val result = runCatching { import() }
            fetching = false; fetched = null
            val done = result.getOrNull()
            notice = when {
                done == null -> context.getString(R.string.prepared_failed)
                done.parts == 0 && done.alreadyThere > 0 -> context.getString(R.string.prepared_nothing_new)
                done.parts == 0 -> context.getString(R.string.prepared_none)
                else -> context.getString(R.string.prepared_done, done.parts)
            }
            if (done != null && done.parts > 0 && done.books.size == 1) onOpen(done.books.single())
        }
    }
    val pickAudio = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) prepared { app.library.importPrepared(uris) }
    }
    fun choose() = pick.launch(arrayOf("application/epub+zip", "text/plain", "application/octet-stream"))

    Scaffold(
        modifier = modifier,
        topBar = { TopAppBar(title = { Text(stringResource(R.string.tab_library)) }) },
        floatingActionButton = {
            // The three ways to add a book: its file, or audio made on a PC, from storage or from a link.
            if (books.isNotEmpty()) Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                ExtendedFloatingActionButton(onClick = { linking = true }, containerColor = MaterialTheme.colorScheme.secondaryContainer,
                    icon = { Icon(painterResource(R.drawable.ic_headphones), null) }, text = { Text(stringResource(R.string.prepared_from_link)) })
                ExtendedFloatingActionButton(onClick = { pickAudio.launch(arrayOf("*/*")) }, containerColor = MaterialTheme.colorScheme.secondaryContainer,
                    icon = { Icon(painterResource(R.drawable.ic_headphones), null) }, text = { Text(stringResource(R.string.prepared_files)) })
                ExtendedFloatingActionButton(onClick = ::choose, icon = { Icon(Icons.Filled.Add, null) }, text = { Text(stringResource(R.string.add_book)) })
            }
        },
        bottomBar = {
            val id = narration.bookId
            if (id != null) Card(Modifier.fillMaxWidth().padding(12.dp).clickable { onOpen(id) }) {
                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(narration.title, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.titleSmall)
                        Text(narration.chapterTitle, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodySmall)
                    }
                    IconButton(onClick = { app.narration.toggle() }) {
                        Icon(painterResource(if (narration.playing) R.drawable.ic_pause else R.drawable.ic_play), stringResource(if (narration.playing) R.string.pause else R.string.play))
                    }
                }
            }
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            if (importing || fetching) LinearProgressIndicator(Modifier.fillMaxWidth())
            fetched?.let { (done, total) -> Text(stringResource(R.string.prepared_progress, done, total), Modifier.padding(horizontal = 16.dp, vertical = 4.dp), style = MaterialTheme.typography.labelMedium) }
            if (books.isEmpty()) {
                Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(painterResource(R.drawable.ic_notification), null, Modifier.size(64.dp), tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.height(16.dp))
                    Text(stringResource(R.string.library_empty), style = MaterialTheme.typography.bodyLarge)
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = ::choose, enabled = !importing) { Text(stringResource(R.string.add_book)) }
                    OutlinedButton(onClick = { pickAudio.launch(arrayOf("*/*")) }, enabled = !fetching) { Text(stringResource(R.string.prepared_files)) }
                    OutlinedButton(onClick = { linking = true }, enabled = !fetching) { Text(stringResource(R.string.prepared_from_link)) }
                }
            } else LazyColumn(contentPadding = PaddingValues(12.dp, 4.dp, 12.dp, 220.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(books, key = { it.id }) { book -> BookCard(book, onOpen) }
            }
        }
    }
    if (linking) LinkDialog(onDismiss = { linking = false }) { links -> linking = false; prepared { app.library.importPreparedLinks(links) { done, total -> fetched = done to total } } }
    notice?.let { message ->
        AlertDialog(onDismissRequest = { notice = null }, text = { Text(message) }, confirmButton = { TextButton({ notice = null }) { Text(stringResource(R.string.close)) } })
    }
}

@Composable
private fun LinkDialog(onDismiss: () -> Unit, onFetch: (List<String>) -> Unit) {
    var text by remember { mutableStateOf("") }
    val links = text.lines().map { it.trim() }.filter { it.startsWith("https://") || it.startsWith("http://") }
    AlertDialog(onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.prepared_link)) },
        text = {
            Column {
                Text(stringResource(R.string.prepared_link_hint), style = MaterialTheme.typography.bodySmall)
                OutlinedTextField(text, { text = it }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp), minLines = 3, maxLines = 8, placeholder = { Text("https://…") })
            }
        },
        confirmButton = { TextButton({ onFetch(links) }, enabled = links.isNotEmpty()) { Text(stringResource(R.string.download)) } },
        dismissButton = { TextButton(onDismiss) { Text(stringResource(R.string.cancel)) } })
}

@Composable
private fun BookCard(book: BookSummary, onOpen: (String) -> Unit) {
    val app = ReaderApp.get(LocalContext.current)
    val scope = rememberCoroutineScope()
    var menu by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    Card(Modifier.fillMaxWidth().clickable { onOpen(book.id) }) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(book.title, style = MaterialTheme.typography.titleMedium.copy(fontFamily = FontFamily.Serif), maxLines = 2, overflow = TextOverflow.Ellipsis)
                book.author?.let { Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                Spacer(Modifier.height(8.dp))
                val percent = if (book.totalPieces > 0) book.piece * 100 / book.totalPieces else 0
                Text(stringResource(R.string.book_progress, book.chapter + 1, book.chapterTitles.size, percent), style = MaterialTheme.typography.labelMedium)
                LinearProgressIndicator(progress = { percent / 100f }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
            }
            Box {
                IconButton(onClick = { menu = true }) { Icon(Icons.Filled.MoreVert, stringResource(R.string.more)) }
                DropdownMenu(menu, { menu = false }) {
                    DropdownMenuItem(text = { Text(stringResource(R.string.clear_audio)) }, onClick = { menu = false; scope.launch { app.library.clearAudio(book.id) } })
                    DropdownMenuItem(text = { Text(stringResource(R.string.delete_book)) }, onClick = { menu = false; confirmDelete = true })
                }
            }
        }
    }
    if (confirmDelete) AlertDialog(
        onDismissRequest = { confirmDelete = false },
        title = { Text(stringResource(R.string.delete_book)) },
        text = { Text(stringResource(R.string.delete_book_confirm, book.title)) },
        confirmButton = { TextButton({ confirmDelete = false; if (app.narration.state.value.bookId == book.id) app.narration.close(); scope.launch { app.library.delete(book.id) } }) { Text(stringResource(R.string.delete)) } },
        dismissButton = { TextButton({ confirmDelete = false }) { Text(stringResource(R.string.cancel)) } },
    )
}
