import SwiftUI

struct ContentView: View {
    @StateObject private var speechRecognizer = SpeechRecognizer()

    var body: some View {
        VStack(spacing: 20) {
            Text("Offline Transcriber")
                .font(.title)
                .padding()

            if let errorMessage = speechRecognizer.errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding()
            }

            ScrollView {
                Text(speechRecognizer.transcript)
                    .font(.body)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            .padding()

            Button(action: {
                if speechRecognizer.isRecording {
                    speechRecognizer.stopTranscribing()
                } else {
                    speechRecognizer.startTranscribing()
                }
            }) {
                Text(speechRecognizer.isRecording ? "Stop Recording" : "Start Recording")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(speechRecognizer.isRecording ? Color.red : Color.blue)
                    .cornerRadius(10)
            }
            .padding()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
