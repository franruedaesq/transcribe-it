import SwiftUI

struct ContentView: View {
    @StateObject private var appleRecognizer = SpeechRecognizer()
    @StateObject private var mlxRecognizer = MLXSpeechRecognizer()

    enum RecognizerType {
        case apple
        case mlx
    }

    @State private var selectedRecognizer: RecognizerType = .apple

    var body: some View {
        VStack(spacing: 20) {
            Text("Offline Transcriber")
                .font(.title)
                .padding()

            Picker("Recognizer", selection: $selectedRecognizer) {
                Text("Apple Speech").tag(RecognizerType.apple)
                Text("Qwen3-ASR (MLX)").tag(RecognizerType.mlx)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal)

            if let errorMessage = currentErrorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding()
            }

            if selectedRecognizer == .mlx && !mlxRecognizer.isModelLoaded {
                Text("Model weights not loaded. Please see AGENTS.md.")
                    .font(.caption)
                    .foregroundColor(.gray)
            }

            ScrollView {
                Text(currentTranscript)
                    .font(.body)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            .padding()

            if selectedRecognizer == .mlx && mlxRecognizer.isProcessing {
                ProgressView("Transcribing...")
            }

            Button(action: toggleRecording) {
                Text(isRecording ? "Stop Recording" : "Start Recording")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(buttonColor)
                    .cornerRadius(10)
            }
            .padding()
            .disabled(isButtonDisabled)
        }
    }

    // MARK: - Helpers

    private var isRecording: Bool {
        switch selectedRecognizer {
        case .apple: return appleRecognizer.isRecording
        case .mlx: return mlxRecognizer.isRecording
        }
    }

    private var currentTranscript: String {
        switch selectedRecognizer {
        case .apple: return appleRecognizer.transcript
        case .mlx: return mlxRecognizer.transcript
        }
    }

    private var currentErrorMessage: String? {
        switch selectedRecognizer {
        case .apple: return appleRecognizer.errorMessage
        case .mlx: return mlxRecognizer.errorMessage
        }
    }

    private var buttonColor: Color {
        if isButtonDisabled { return .gray }
        return isRecording ? .red : .blue
    }

    private var isButtonDisabled: Bool {
        if selectedRecognizer == .mlx {
            return !mlxRecognizer.isModelLoaded || mlxRecognizer.isProcessing
        }
        return false
    }

    private func toggleRecording() {
        switch selectedRecognizer {
        case .apple:
            if appleRecognizer.isRecording {
                appleRecognizer.stopTranscribing()
            } else {
                appleRecognizer.startTranscribing()
            }
        case .mlx:
            if mlxRecognizer.isRecording {
                mlxRecognizer.stopRecording()
            } else {
                mlxRecognizer.startRecording()
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
