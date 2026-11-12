package edu.aub.focuspolice.watch

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.app.ServiceCompat
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger

/** Live values the screen reads twice a second. */
object Status {
    @Volatile var running = false
    @Volatile var session = ""
    @Volatile var rate = 0
    @Volatile var sent = 0L
    @Volatile var queued = 0
    @Volatile var lastOkMs = 0L
    @Volatile var message = "Idle"
    @Volatile var network = "-"
    @Volatile var missing = ""
}

/**
 * Records wrist motion at 50 Hz as a foreground service (keeps running with the screen off),
 * writes every sample to a CSV on the watch, and POSTs 1-second batches to the laptop.
 * If the laptop can't be reached, samples queue up (30 min) and are sent when it's back.
 */
class SensorService : Service(), SensorEventListener {

    companion object {
        const val ACTION_START = "edu.aub.focuspolice.watch.START"
        const val ACTION_STOP = "edu.aub.focuspolice.watch.STOP"
        const val HZ = 50
        const val G = 9.80665
        const val MAX_QUEUE = HZ * 60 * 30
        const val BATCH = HZ * 30
        const val CHANNEL = "focus"
        val COLS = listOf(
            "time", "userAccelerationX", "userAccelerationY", "userAccelerationZ",
            "gravityX", "gravityY", "gravityZ", "rotationRateX", "rotationRateY", "rotationRateZ",
            "roll", "pitch", "yaw", "accX", "accY", "accZ"
        )
        fun prefs(c: Context) = c.getSharedPreferences("focus", Context.MODE_PRIVATE)
        fun server(c: Context): String = prefs(c).getString("server", "http://192.168.0.10:8000")!!
    }

    private lateinit var sm: SensorManager
    private var sensorThread: HandlerThread? = null
    private val lin = FloatArray(3)
    private val grav = FloatArray(3)
    private val gyro = FloatArray(3)
    private val rotM = FloatArray(9)
    private val ori = FloatArray(3)

    private class Row(val seq: Long, val v: DoubleArray)
    private val queue = ArrayDeque<Row>()
    private var seq = 0L
    private val counter = AtomicInteger(0)
    private var writer: BufferedWriter? = null
    private var linesSinceFlush = 0
    private var epochOffset = 0.0

    @Volatile private var stopping = false
    private var uploader: Thread? = null
    private var wake: PowerManager.WakeLock? = null
    @Volatile private var wifi: Network? = null
    private var netCb: ConnectivityManager.NetworkCallback? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> { stopRecording(); return START_NOT_STICKY }
            ACTION_START -> startRecording()
            null -> {   // restarted by the system after being killed: resume if we were recording
                if (prefs(this).getBoolean("recording", false)) startRecording() else stopSelf()
            }
        }
        return START_STICKY
    }

    // ---------------------------------------------------------------- start

    private fun startRecording() {
        if (Status.running) return
        try {
            ServiceCompat.startForeground(
                this, 1, buildNotification(),
                if (Build.VERSION.SDK_INT >= 34) ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH else 0
            )
        } catch (e: Exception) {
            Status.message = "Could not start in background: ${e.javaClass.simpleName}"
            stopSelf(); return
        }
        prefs(this).edit().putBoolean("recording", true).apply()
        stopping = false

        wake = (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "FocusPolice:record").apply { acquire() }

        val id = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val dir = File(filesDir, "sessions").apply { mkdirs() }
        writer = BufferedWriter(FileWriter(File(dir, "$id.csv")), 1 shl 16).apply {
            write(COLS.joinToString(",")); newLine()
        }
        epochOffset = System.currentTimeMillis() / 1000.0 - SystemClock.elapsedRealtimeNanos() / 1e9
        synchronized(queue) { queue.clear() }

        sm = getSystemService(SENSOR_SERVICE) as SensorManager
        val th = HandlerThread("sensors").apply { start() }
        sensorThread = th
        val handler = Handler(th.looper)
        val period = 1_000_000 / HZ
        val missing = mutableListOf<String>()
        for ((type, name) in listOf(
            Sensor.TYPE_LINEAR_ACCELERATION to "linear", Sensor.TYPE_GRAVITY to "gravity",
            Sensor.TYPE_GYROSCOPE to "gyro", Sensor.TYPE_ROTATION_VECTOR to "rotation",
            Sensor.TYPE_ACCELEROMETER to "accel"   // accelerometer last: it drives the rows
        )) {
            val s = sm.getDefaultSensor(type)
            if (s == null || !sm.registerListener(this, s, period, 0, handler)) missing += name
        }
        Status.missing = missing.joinToString(",")

        requestWifi()

        Status.session = id
        Status.sent = 0; Status.queued = 0; Status.rate = 0; Status.lastOkMs = 0
        Status.message = if ("accel" in missing) "NO ACCELEROMETER" else "Recording…"
        Status.running = true

        uploader = Thread({ uploadLoop() }, "uploader").apply { start() }
    }

    // ---------------------------------------------------------------- sensors

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> System.arraycopy(e.values, 0, lin, 0, 3)
            Sensor.TYPE_GRAVITY -> System.arraycopy(e.values, 0, grav, 0, 3)
            Sensor.TYPE_GYROSCOPE -> System.arraycopy(e.values, 0, gyro, 0, 3)
            Sensor.TYPE_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rotM, e.values)
                SensorManager.getOrientation(rotM, ori)   // [yaw, pitch, roll]
            }
            Sensor.TYPE_ACCELEROMETER -> {
                val t = epochOffset + e.timestamp / 1e9
                val v = doubleArrayOf(
                    t,
                    lin[0] / G, lin[1] / G, lin[2] / G,
                    grav[0] / G, grav[1] / G, grav[2] / G,
                    gyro[0].toDouble(), gyro[1].toDouble(), gyro[2].toDouble(),
                    ori[2].toDouble(), ori[1].toDouble(), ori[0].toDouble(),
                    e.values[0] / G, e.values[1] / G, e.values[2] / G
                )
                synchronized(queue) {
                    queue.addLast(Row(seq++, v))
                    if (queue.size > MAX_QUEUE) queue.removeFirst()   // still on disk
                }
                writer?.let { w ->
                    val sb = StringBuilder(160)
                    sb.append(String.format(Locale.US, "%.4f", v[0]))
                    for (i in 1 until v.size) sb.append(',').append(String.format(Locale.US, "%.5f", v[i]))
                    w.write(sb.toString()); w.newLine()
                    if (++linesSinceFlush >= HZ) { w.flush(); linesSinceFlush = 0 }
                }
                counter.incrementAndGet()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ---------------------------------------------------------------- network

    private fun requestWifi() {
        val cm = getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager
        val cb = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) { wifi = network }
            override fun onLost(network: Network) { if (wifi == network) wifi = null }
        }
        netCb = cb
        try {
            // On Wear OS this asks the system to bring Wi-Fi up and keep it up for us
            // (otherwise the watch may route through a phone or turn Wi-Fi off to save power).
            cm.requestNetwork(
                NetworkRequest.Builder().addTransportType(NetworkCapabilities.TRANSPORT_WIFI).build(), cb
            )
        } catch (e: Exception) { netCb = null }
    }

    private fun uploadLoop() {
        var lastTick = SystemClock.elapsedRealtime()
        var stopDeadline = 0L
        while (true) {
            try { Thread.sleep(250) } catch (_: InterruptedException) {}
            val now = SystemClock.elapsedRealtime()
            if (now - lastTick >= 1000) {
                Status.rate = (counter.getAndSet(0) * 1000 / (now - lastTick)).toInt()
                lastTick = now
            }
            if (stopping) {
                if (stopDeadline == 0L) stopDeadline = now + 8000
                if (now > stopDeadline) break
            }
            val batch: List<Row> = synchronized(queue) { queue.take(BATCH) }
            Status.queued = synchronized(queue) { queue.size }
            if (batch.isEmpty()) { if (stopping) break else continue }
            // send at most once a second unless catching up on a backlog
            if (!stopping && batch.size < HZ && now - Status.lastOkMs < 1000) continue

            if (post(batch)) {
                val last = batch.last().seq
                synchronized(queue) { while (queue.isNotEmpty() && queue.first().seq <= last) queue.removeFirst() }
                Status.sent += batch.size
                Status.lastOkMs = SystemClock.elapsedRealtime()
                Status.message = "Streaming"
            } else {
                try { Thread.sleep(1500) } catch (_: InterruptedException) {}
            }
        }
    }

    private fun post(batch: List<Row>): Boolean {
        val body = StringBuilder(batch.size * 200 + 300)
        body.append("{\"session\":\"").append(Status.session)
            .append("\",\"device\":\"").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
            .append("\",\"os\":\"Android ").append(Build.VERSION.RELEASE)
            .append("\",\"hz\":").append(HZ).append(",\"cols\":[")
        COLS.forEachIndexed { i, c -> if (i > 0) body.append(','); body.append('"').append(c).append('"') }
        body.append("],\"rows\":[")
        batch.forEachIndexed { i, r ->
            if (i > 0) body.append(',')
            body.append('[')
            r.v.forEachIndexed { j, x -> if (j > 0) body.append(','); body.append(if (x.isFinite()) x.toString() else "0") }
            body.append(']')
        }
        body.append("]}")
        val bytes = body.toString().toByteArray()
        val url = URL(server(this).trimEnd('/') + "/ingest")
        // try Wi-Fi first, then whatever network the system offers
        for (net in listOf(wifi, null)) {
            try {
                val conn = (net?.openConnection(url) ?: url.openConnection()) as HttpURLConnection
                conn.connectTimeout = 4000; conn.readTimeout = 5000
                conn.requestMethod = "POST"; conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json")
                conn.setFixedLengthStreamingMode(bytes.size)
                conn.outputStream.use { it.write(bytes) }
                val code = conn.responseCode
                conn.inputStream.use { it.readBytes() }
                conn.disconnect()
                if (code == 200) { Status.network = if (net != null) "Wi-Fi" else "default"; return true }
                Status.message = "Laptop answered HTTP $code"
            } catch (e: Exception) {
                Status.message = "No laptop at ${server(this)} (${e.javaClass.simpleName})"
            }
            if (net == null) break
        }
        return false
    }

    // ---------------------------------------------------------------- stop

    private fun stopRecording() {
        if (!Status.running) { stopSelf(); return }
        Status.running = false
        Status.message = "Stopping… sending the rest"
        prefs(this).edit().putBoolean("recording", false).apply()
        try { sm.unregisterListener(this) } catch (_: Exception) {}
        Thread {
            sensorThread?.quitSafely()
            try { sensorThread?.join(2000) } catch (_: Exception) {}
            try { writer?.flush(); writer?.close() } catch (_: Exception) {}
            writer = null
            stopping = true
            try { uploader?.join(10_000) } catch (_: Exception) {}
            netCb?.let { try { (getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager).unregisterNetworkCallback(it) } catch (_: Exception) {} }
            netCb = null; wifi = null
            try { if (wake?.isHeld == true) wake?.release() } catch (_: Exception) {}
            val left = synchronized(queue) { queue.size }
            Status.queued = left
            Status.message = if (left == 0) "Stopped. All sent." else "Stopped. $left samples not sent (saved on watch)"
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf()
        }.start()
    }

    override fun onDestroy() {
        if (Status.running) {  // killed without Stop: keep what we have
            try { sm.unregisterListener(this) } catch (_: Exception) {}
            try { writer?.flush(); writer?.close() } catch (_: Exception) {}
            Status.running = false
        }
        super.onDestroy()
    }

    // ---------------------------------------------------------------- notification

    private fun buildNotification(): Notification {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Recording", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, SensorService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("Focus Police")
            .setContentText("Recording wrist motion")
            .setSmallIcon(R.drawable.ic_launcher)
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }
}
