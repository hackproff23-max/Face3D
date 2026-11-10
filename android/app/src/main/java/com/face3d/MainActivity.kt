package com.face3d

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import coil.compose.AsyncImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.math.max

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(Modifier.fillMaxSize()) { App() }
            }
        }
    }
}

private val labels = listOf("Frente", "Lado esquerdo", "Lado direito")

@Composable
fun App() {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val prefs = remember { ctx.getSharedPreferences("cfg", Context.MODE_PRIVATE) }
    var server by remember { mutableStateOf(prefs.getString("server", "http://192.168.0.10:8000") ?: "") }
    val uris = remember { mutableStateListOf<Uri?>(null, null, null) }
    var pick by remember { mutableIntStateOf(0) }
    var busy by remember { mutableStateOf(false) }
    var msg by remember { mutableStateOf("") }
    var glb by remember { mutableStateOf<ByteArray?>(null) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { u ->
        if (u != null) uris[pick] = u
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("Rosto 3D (GLB com esqueleto)", style = MaterialTheme.typography.titleLarge)
        OutlinedTextField(
            value = server,
            onValueChange = { server = it; prefs.edit().putString("server", it).apply() },
            label = { Text("URL do servidor") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Text("Toque nos quadros: frente, perfil esquerdo e perfil direito. Luz uniforme, sem óculos.",
            style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (i in 0..2) {
                Box(
                    Modifier.weight(1f).aspectRatio(0.75f).clip(RoundedCornerShape(12.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                        .clickable { pick = i; picker.launch("image/*") },
                    contentAlignment = Alignment.Center
                ) {
                    val u = uris[i]
                    if (u != null) AsyncImage(u, labels[i], Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
                    else Text(labels[i])
                }
            }
        }
        Button(
            enabled = !busy && uris.all { it != null } && server.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
            onClick = {
                busy = true; glb = null; msg = "Enviando e gerando… (pode levar ~1 min)"
                scope.launch {
                    try {
                        val bytes = withContext(Dispatchers.IO) { generate(ctx, server, uris.map { it!! }) }
                        glb = bytes; msg = "Pronto! ${bytes.size / 1024} KB"
                    } catch (e: Exception) {
                        msg = "Erro: ${e.message}"
                    }
                    busy = false
                }
            }
        ) { Text(if (busy) "Gerando…" else "Gerar modelo 3D") }
        if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
        if (msg.isNotEmpty()) Text(msg)
        glb?.let { data ->
            Button(onClick = { msg = saveDownloads(ctx, data) }, modifier = Modifier.fillMaxWidth()) {
                Text("Salvar em Downloads")
            }
            OutlinedButton(onClick = { share(ctx, data) }, modifier = Modifier.fillMaxWidth()) {
                Text("Abrir / compartilhar .glb")
            }
        }
    }
}

private val http = OkHttpClient.Builder()
    .connectTimeout(30, TimeUnit.SECONDS)
    .writeTimeout(120, TimeUnit.SECONDS)
    .readTimeout(300, TimeUnit.SECONDS)
    .build()

fun generate(ctx: Context, server: String, uris: List<Uri>): ByteArray {
    val names = listOf("front", "left", "right")
    val mb = MultipartBody.Builder().setType(MultipartBody.FORM)
    uris.forEachIndexed { i, u ->
        mb.addFormDataPart(names[i], "${names[i]}.jpg", loadJpeg(ctx, u).toRequestBody("image/jpeg".toMediaType()))
    }
    val req = Request.Builder().url(server.trim().trimEnd('/') + "/generate").post(mb.build()).build()
    http.newCall(req).execute().use { r ->
        val body = r.body?.bytes() ?: ByteArray(0)
        if (!r.isSuccessful) error(String(body).take(300).ifEmpty { "HTTP ${r.code}" })
        return body
    }
}

fun loadJpeg(ctx: Context, uri: Uri): ByteArray {
    val cr = ctx.contentResolver
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    cr.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
    var s = 1
    while (max(bounds.outWidth, bounds.outHeight) / s > 2000) s *= 2
    var bmp = cr.openInputStream(uri)!!.use {
        BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = s })
    } ?: error("Imagem inválida")
    val orient = cr.openInputStream(uri)!!.use {
        ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    }
    val m = Matrix()
    when (orient) {
        ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
        ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
        ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
    }
    if (!m.isIdentity) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    return ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }.toByteArray()
}

fun saveDownloads(ctx: Context, data: ByteArray): String = try {
    val v = ContentValues().apply {
        put(MediaStore.Downloads.DISPLAY_NAME, "rosto_${System.currentTimeMillis()}.glb")
        put(MediaStore.Downloads.MIME_TYPE, "model/gltf-binary")
        put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
    }
    val uri = ctx.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v)!!
    ctx.contentResolver.openOutputStream(uri)!!.use { it.write(data) }
    "Salvo em Downloads"
} catch (e: Exception) {
    "Erro ao salvar: ${e.message}"
}

fun share(ctx: Context, data: ByteArray) {
    val f = File(ctx.cacheDir, "rosto.glb").also { it.writeBytes(data) }
    val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", f)
    val i = Intent(Intent.ACTION_SEND).apply {
        type = "model/gltf-binary"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(i, "Abrir GLB"))
}
