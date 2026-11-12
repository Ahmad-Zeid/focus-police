import Foundation
import CoreMotion
import WatchKit

/// Reads wrist motion at a fixed rate and POSTs it to the laptop every second.
/// A physical-therapy extended runtime session keeps the app alive with the screen off (max 1 h).
final class Streamer: NSObject, ObservableObject, WKExtendedRuntimeSessionDelegate {
    // Change in the app (Server field) or here.
    @Published var server: String = UserDefaults.standard.string(forKey: "server") ?? "http://192.168.0.10:8000" {
        didSet { UserDefaults.standard.set(server, forKey: "server") }
    }
    @Published var running = false
    @Published var status = "Idle"
    @Published var sent = 0
    @Published var buffered = 0
    @Published var rate = 0.0

    let hz = 50.0
    private let motion = CMMotionManager()
    private let queue = OperationQueue()
    private var runtime: WKExtendedRuntimeSession?
    private var timer: Timer?
    private var pending: [[Double]] = []          // rows not yet delivered
    private let lock = NSLock()
    private var inFlight = false
    private var sessionID = ""
    private var uptimeToEpoch = 0.0
    private var countThisSecond = 0
    private let http: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 4
        c.waitsForConnectivity = false
        return URLSession(configuration: c)
    }()

    // MARK: start / stop

    func start() {
        guard !running else { return }
        guard motion.isDeviceMotionAvailable else { status = "No device motion"; return }
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
        sessionID = f.string(from: Date())
        uptimeToEpoch = Date().timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime
        lock.lock(); pending.removeAll(); lock.unlock()
        sent = 0; buffered = 0

        let s = WKExtendedRuntimeSession()
        s.delegate = self
        s.start()
        runtime = s

        queue.maxConcurrentOperationCount = 1
        motion.deviceMotionUpdateInterval = 1.0 / hz
        motion.startDeviceMotionUpdates(to: queue) { [weak self] dm, _ in
            guard let self = self, let dm = dm else { return }
            let ua = dm.userAcceleration, g = dm.gravity, r = dm.rotationRate, a = dm.attitude
            let row: [Double] = [
                self.uptimeToEpoch + dm.timestamp,           // epoch seconds
                ua.x, ua.y, ua.z, g.x, g.y, g.z,             // g units
                r.x, r.y, r.z,                               // rad/s
                a.roll, a.pitch, a.yaw                       // rad
            ]
            self.lock.lock(); self.pending.append(row)
            if self.pending.count > 50 * 60 * 10 { self.pending.removeFirst(self.pending.count - 50 * 60 * 10) } // keep max 10 min
            self.lock.unlock()
            self.countThisSecond += 1
        }

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in self?.tick() }
        running = true
        status = "Starting…"
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
        timer?.invalidate(); timer = nil
        flush()                       // last batch
        runtime?.invalidate(); runtime = nil
        running = false
        status = "Stopped"
    }

    // MARK: sending

    private func tick() {
        rate = Double(countThisSecond); countThisSecond = 0
        flush()
    }

    private func flush() {
        guard !inFlight, let url = URL(string: server + "/ingest") else { return }
        lock.lock()
        let batch = Array(pending.prefix(50 * 30))   // at most 30 s of backlog per request
        lock.unlock()
        buffered = pending.count
        guard !batch.isEmpty else { return }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["session": sessionID, "device": WKInterfaceDevice.current().model,
                                   "hz": hz, "rows": batch]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        inFlight = true
        http.dataTask(with: req) { [weak self] _, resp, err in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.inFlight = false
                if let code = (resp as? HTTPURLResponse)?.statusCode, code == 200 {
                    self.lock.lock(); self.pending.removeFirst(min(batch.count, self.pending.count)); self.lock.unlock()
                    self.sent += batch.count
                    self.buffered = self.pending.count
                    self.status = "Streaming"
                } else {
                    self.status = "No laptop: " + (err?.localizedDescription ?? "HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0)")
                }
            }
        }.resume()
    }

    // MARK: runtime session delegate

    func extendedRuntimeSessionDidStart(_ s: WKExtendedRuntimeSession) {
        DispatchQueue.main.async { if self.status == "Starting…" { self.status = "Running" } }
    }
    func extendedRuntimeSessionWillExpire(_ s: WKExtendedRuntimeSession) {
        DispatchQueue.main.async { self.status = "Session ending soon (1 h limit)" }
    }
    func extendedRuntimeSession(_ s: WKExtendedRuntimeSession,
                                didInvalidateWith reason: WKExtendedRuntimeSessionInvalidationReason, error: Error?) {
        DispatchQueue.main.async {
            if self.running { self.status = "Background session ended (\(reason.rawValue)) — keep screen on" }
        }
    }
}
