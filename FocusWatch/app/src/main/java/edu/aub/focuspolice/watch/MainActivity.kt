package edu.aub.focuspolice.watch

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** One screen: Start/Stop, live status, and the laptop address. */
class MainActivity : Activity() {

    private lateinit var status: TextView
    private lateinit var button: Button
    private lateinit var server: EditText
    private val ui = Handler(Looper.getMainLooper())
    private val tick = object : Runnable {
        override fun run() { refresh(); ui.postDelayed(this, 500) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        handleExtras(intent)
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleExtras(intent)
    }

    /** Lets the setup script (adb) set the laptop address and start/stop remotely. */
    private fun handleExtras(i: Intent?) {
        i ?: return
        i.getStringExtra("server")?.let { saveServer(it) }
        if (i.getBooleanExtra("autostart", false) && !Status.running) start()
        if (i.getBooleanExtra("autostop", false) && Status.running) stop()
    }

    override fun onResume() { super.onResume(); ui.post(tick) }
    override fun onPause() { super.onPause(); ui.removeCallbacks(tick) }

    private fun start() {
        startForegroundService(Intent(this, SensorService::class.java).setAction(SensorService.ACTION_START))
    }

    private fun stop() {
        startService(Intent(this, SensorService::class.java).setAction(SensorService.ACTION_STOP))
    }

    private fun saveServer(s: String) {
        var v = s.trim()
        if (v.isEmpty()) return
        // "192.168.0.10" -> http://192.168.0.10:8000 ; "https://example.com" stays as typed (port 443)
        if (!v.startsWith("http")) v = "http://$v"
        val host = v.substringAfter("://").substringBefore("/")
        if (v.startsWith("http://") && !host.contains(":")) v = v.replaceFirst(host, "$host:8000")
        v = v.trimEnd('/')
        SensorService.prefs(this).edit().putString("server", v).apply()
        if (::server.isInitialized) server.setText(v)
    }

    private fun refresh() {
        val running = Status.running
        button.text = if (running) "STOP" else "START"
        button.setBackgroundColor(if (running) Color.rgb(179, 38, 30) else Color.rgb(47, 107, 58))
        val age = if (Status.lastOkMs == 0L) -1L else (SystemClock.elapsedRealtime() - Status.lastOkMs) / 1000
        val link = when {
            !running -> ""
            age in 0..3 -> "● laptop connected (${Status.network})"
            else -> "○ laptop NOT reached"
        }
        status.text = buildString {
            append(Status.message)
            if (running) {
                append("\n").append(link)
                append("\n${Status.rate} Hz · sent ${Status.sent}")
                if (Status.queued > SensorService.HZ * 2) append("\nwaiting to send: ${Status.queued}")
                if (Status.missing.isNotEmpty()) append("\nmissing sensors: ${Status.missing}")
            }
            if (Status.session.isNotEmpty()) append("\nsession ${Status.session}")
        }
        status.setTextColor(if (running && age in 0..3) Color.rgb(111, 191, 125) else if (running) Color.rgb(224, 145, 90) else Color.LTGRAY)
        server.isEnabled = !running
    }

    private fun dp(v: Int) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()

    private fun buildUi() {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            // take focus ourselves so the address field never pops the keyboard on open
            isFocusable = true; isFocusableInTouchMode = true
            descendantFocusability = android.view.ViewGroup.FOCUS_BEFORE_DESCENDANTS
            setPadding(dp(22), dp(26), dp(22), dp(40))
        }
        col.addView(TextView(this).apply {
            text = "Focus Police"; setTextColor(Color.WHITE); textSize = 15f
            typeface = Typeface.DEFAULT_BOLD; gravity = Gravity.CENTER
        })
        button = Button(this).apply {
            textSize = 18f; setTextColor(Color.WHITE)
            setOnClickListener { if (Status.running) stop() else start() }
        }
        col.addView(button, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(56)).apply { topMargin = dp(6) })
        status = TextView(this).apply { textSize = 12f; gravity = Gravity.CENTER; setPadding(0, dp(6), 0, dp(6)) }
        col.addView(status)
        col.addView(TextView(this).apply { text = "Laptop address"; textSize = 11f; setTextColor(Color.GRAY); gravity = Gravity.CENTER })
        server = EditText(this).apply {
            setText(SensorService.server(this@MainActivity)); textSize = 12f; setTextColor(Color.WHITE)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            imeOptions = EditorInfo.IME_ACTION_DONE; isSingleLine = true; gravity = Gravity.CENTER
            setOnEditorActionListener { v, _, _ -> saveServer(v.text.toString()); false }
            setOnFocusChangeListener { v, has -> if (!has) saveServer((v as EditText).text.toString()) }
        }
        col.addView(server, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        setContentView(ScrollView(this).apply { isFillViewport = true; addView(col); setBackgroundColor(Color.BLACK) })
        window.decorView.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_AUTO
    }
}
