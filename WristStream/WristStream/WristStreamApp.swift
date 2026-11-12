import SwiftUI

@main
struct WristStreamApp: App {
    @StateObject private var streamer = Streamer()
    var body: some Scene {
        WindowGroup { ContentView().environmentObject(streamer) }
    }
}

struct ContentView: View {
    @EnvironmentObject var s: Streamer
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 6) {
                Button(s.running ? "Stop" : "Start") { s.running ? s.stop() : s.start() }
                    .tint(s.running ? .red : .green)
                Text(s.status).font(.footnote).foregroundColor(s.status == "Streaming" ? .green : .orange)
                Text("\(Int(s.rate)) Hz · sent \(s.sent)").font(.footnote)
                if s.buffered > 100 { Text("waiting: \(s.buffered)").font(.footnote).foregroundColor(.orange) }
                Text("Server").font(.caption2).foregroundColor(.gray)
                TextField("http://IP:8000", text: $s.server).font(.footnote).disabled(s.running)
            }
        }
    }
}
