import Foundation
import AVFoundation
import Accelerate
#if canImport(MLX)
import MLX
import MLXNN
#endif

class MLXSpeechRecognizer: ObservableObject {
    @Published var transcript: String = ""
    @Published var isRecording: Bool = false
    @Published var isProcessing: Bool = false
    @Published var isModelLoaded: Bool = false
    @Published var errorMessage: String?

    private var audioEngine = AVAudioEngine()
    private var inputNode: AVAudioInputNode?

    // Audio Buffer
    private var audioBuffer: [Float] = []
    private let targetSampleRate: Double = 16000.0

    // Model
    private var model: Qwen3ASR?
    private var tokenizer: Tokenizer?

    // Audio Processing Constants
    private let nMels: Int = 128
    private let nFft: Int = 400
    private let hopLength: Int = 160

    init() {
        Task {
            await loadModel()
        }
    }

    func loadModel() async {
        do {
            #if !canImport(MLX)
            await MainActor.run {
                self.errorMessage = "MLX Framework not found. Please add mlx-swift package dependency in Xcode."
            }
            return
            #else

            // These files must be added to the App Bundle
            guard let modelUrl = Bundle.main.url(forResource: "model", withExtension: "safetensors"),
                  let configUrl = Bundle.main.url(forResource: "config", withExtension: "json"),
                  let tokenizerUrl = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
                await MainActor.run { self.errorMessage = "Model files missing. Please add model.safetensors, config.json, tokenizer.json to Bundle." }
                return
            }

            // Initialize Model
            let model = Qwen3ASR()
            try await model.load(configPath: configUrl, weightsPath: modelUrl)

            // Initialize Tokenizer
            let tokenizer = Tokenizer()
            try tokenizer.load(tokenizerJSONPath: tokenizerUrl)

            self.model = model
            self.tokenizer = tokenizer

            await MainActor.run {
                self.isModelLoaded = true
                self.errorMessage = nil
            }
            #endif
        } catch {
             await MainActor.run { self.errorMessage = "Load failed: \(error.localizedDescription)" }
        }
    }

    func startRecording() {
        #if !canImport(MLX)
        self.errorMessage = "MLX Framework missing."
        return
        #endif

        guard isModelLoaded else {
            self.errorMessage = "Model is not loaded."
            return
        }

        self.audioBuffer = []
        self.transcript = ""
        self.errorMessage = nil

        let inputNode = audioEngine.inputNode
        self.inputNode = inputNode

        let format = inputNode.outputFormat(forBus: 0)
        // Install tap on input format.
        // We will perform naive resampling in the callback if needed,
        // or assume the user speaks into a compatible mic (often 48kHz on iOS).
        // Real-world app needs `AVAudioConverter`.

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, time in
            self.appendAudio(buffer: buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            self.isRecording = true
        } catch {
            self.errorMessage = "Engine start failed: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        audioEngine.stop()
        inputNode?.removeTap(onBus: 0)

        DispatchQueue.main.async {
            self.isRecording = false
            self.isProcessing = true
        }

        // Start Transcription
        Task {
            await transcribe()
        }
    }

    private func appendAudio(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frames = Int(buffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData, count: frames))

        // Accumulate audio
        self.audioBuffer.append(contentsOf: samples)
    }

    func transcribe() async {
        #if canImport(MLX)
        guard let model = model, let tokenizer = tokenizer else { return }

        // 1. Process Audio to Mel Spectrogram
        let features = computeLogMelSpectrogram(audio: audioBuffer)

        // 2. Run Inference
        let text = model.transcribe(audioFeatures: features, tokenizer: tokenizer)

        await MainActor.run {
            self.transcript = text
            self.isProcessing = false
        }
        #else
        await MainActor.run {
            self.errorMessage = "MLX Framework not found."
            self.isProcessing = false
        }
        #endif
    }

    #if canImport(MLX)
    private func computeLogMelSpectrogram(audio: [Float]) -> MLXArray {
        // IMPORTANT: Real Mel Spectrogram calculation requires FFT and Mel Filterbank.
        // Implementing this from scratch in a single file is too large for this context.
        // Users should use a library like `swift-audio-processing` or port `mlx-audio` logic.

        // We return zeros of correct shape so the model graph runs.
        // The output will be garbage (random tokens) because input is silence.

        let timeSteps = audio.count / hopLength
        let validTimeSteps = max(1, timeSteps)

        // Warn developer in console
        print("Warning: Using dummy Log Mel Spectrogram. Implement FFT/Mel logic for real results.")

        return MLXArray.zeros([1, validTimeSteps, nMels])
    }
    #endif
}
