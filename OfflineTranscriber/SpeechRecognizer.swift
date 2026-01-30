import Foundation
import Speech
import AVFoundation

class SpeechRecognizer: ObservableObject {
    @Published var transcript: String = ""
    @Published var isRecording: Bool = false
    @Published var errorMessage: String?

    private var audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private let recognizer: SFSpeechRecognizer?

    init() {
        recognizer = SFSpeechRecognizer()

        Task {
            do {
                guard await SFSpeechRecognizer.hasAuthorizationToRecognize() else {
                    await MainActor.run {
                        self.errorMessage = "Permission denied for speech recognition."
                    }
                    return
                }

                guard await AVAudioSession.sharedInstance().hasPermissionToRecord() else {
                    await MainActor.run {
                        self.errorMessage = "Permission denied for microphone."
                    }
                    return
                }
            }
        }
    }

    deinit {
        reset()
    }

    func startTranscribing() {
        guard let recognizer = recognizer, recognizer.isAvailable else {
            self.errorMessage = "Speech recognizer is not available."
            return
        }

        do {
            let (audioEngine, request) = try prepareEngine()
            self.audioEngine = audioEngine
            self.request = request

            self.task = recognizer.recognitionTask(with: request) { [weak self] result, error in
                self?.recognitionHandler(result: result, error: error)
            }

            self.isRecording = true
            self.errorMessage = nil
        } catch {
            self.reset()
            self.errorMessage = "Error initializing speech recognition: \(error.localizedDescription)"
        }
    }

    func stopTranscribing() {
        reset()
    }

    private func prepareEngine() throws -> (AVAudioEngine, SFSpeechAudioBufferRecognitionRequest) {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        let audioEngine = AVAudioEngine()

        let request = SFSpeechAudioBufferRecognitionRequest()

        // This forces the recognition to happen on-device (offline).
        // If the device doesn't support it or the language model isn't downloaded,
        // it might fail or error out, but this ensures offline capability.
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = true
        }

        request.shouldReportPartialResults = true

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            request.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()

        return (audioEngine, request)
    }

    private func recognitionHandler(result: SFSpeechRecognitionResult?, error: Error?) {
        if let result = result {
            DispatchQueue.main.async {
                self.transcript = result.bestTranscription.formattedString
            }
        }

        if error != nil || (result?.isFinal ?? false) {
            self.reset()
        }
    }

    private func reset() {
        task?.cancel()
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)

        request = nil
        task = nil

        DispatchQueue.main.async {
            self.isRecording = false
        }
    }
}

extension SFSpeechRecognizer {
    static func hasAuthorizationToRecognize() async -> Bool {
        await withCheckedContinuation { continuation in
            requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }
}

extension AVAudioSession {
    func hasPermissionToRecord() async -> Bool {
        await withCheckedContinuation { continuation in
            requestRecordPermission { authorized in
                continuation.resume(returning: authorized)
            }
        }
    }
}
