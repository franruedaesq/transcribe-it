# Developer Instructions: Qwen3-ASR on iOS with MLX

This repository contains an implementation of the Qwen3-ASR model architecture using `mlx-swift`.
It is designed to run offline on iOS devices with Apple Silicon.

## Status of Implementation

**This implementation is a functional prototype.**
It provides the full model architecture (Transformer Encoder/Decoder), inference loop, and UI integration.
However, due to the complexity of certain components, the following parts are simplified or require external libraries for production use:

1.  **Feature Extraction (Mel Spectrogram)**:
    *   The `MLXSpeechRecognizer.swift` contains a placeholder `computeLogMelSpectrogram` function.
    *   **Action Required**: You must implement a real Mel Spectrogram calculation (STFT -> Mel Filterbank -> Log10) using `Accelerate` (vDSP) or a library like `AudioKit` or port the logic from `mlx-audio` (Python).
    *   Currently, it returns zeros, so the model will output random tokens or silence.

2.  **Tokenizer**:
    *   The `Tokenizer.swift` provides a basic JSON loader for vocab.
    *   **Action Required**: For accurate text decoding, you should use a Swift BPE tokenizer library (e.g., `swift-transformers` or `HuggingFace/Tokenizers` wrapper) that supports `tiktoken` or `BPE` used by Qwen.
    *   The provided implementation does simple character mapping which is insufficient for Qwen's BPE vocabulary.

## Setup Instructions

### 1. Add `mlx-swift` Dependency

1.  Open Xcode.
2.  **File > Add Packages...**.
3.  URL: `https://github.com/ml-explore/mlx-swift`.
4.  Add to `OfflineTranscriber` target.

### 2. Download Model Files

You must download the model files manually as they are large.

1.  Go to [Hugging Face: mlx-community/Qwen3-ASR-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit).
2.  Download:
    *   `model.safetensors`
    *   `config.json`
    *   `tokenizer.json`
3.  Drag and drop them into the `OfflineTranscriber` group in Xcode.
4.  **Important**: Ensure "Add to targets" is checked for `OfflineTranscriber`.

### 3. Build and Run

1.  Select a physical iOS device (recommended for ML performance).
2.  Run the app.
3.  Select "Qwen3-ASR (MLX)" from the picker.
4.  If files are missing, an error will appear.
5.  Press "Start Recording".

## Troubleshooting

*   **"Model files missing"**: Verify the files are in the "Copy Bundle Resources" build phase.
*   **Garbage Output**: This is expected until Feature Extraction is implemented. The model is running on silence/zeros.
*   **Crash on Load**: Check `config.json` compatibility. The code assumes standard Qwen3 config structure.

## Architecture Overview

*   `Qwen3ASR.swift`:
    *   `Qwen3ASRAudioEncoder`: Transformer Encoder (Whisper-like) with `Conv1d` input.
    *   `Qwen3TextDecoder`: Qwen3 (Llama-like) Transformer Decoder.
    *   `Attention`: Implements Scaled Dot Product Attention with RoPE.
    *   `transcribe()`: Implements Greedy Decoding loop.
*   `MLXSpeechRecognizer.swift`: Manages `AVAudioEngine` and Model Inference.
