# Offline iOS Transcriber

This is a SwiftUI application that demonstrates how to perform offline speech transcription on iOS using two methods:
1.  **Apple's Native `Speech` Framework** (`SFSpeechRecognizer`).
2.  **Qwen3-ASR Model** via the `mlx-swift` library.

## Overview

The app allows you to toggle between the native Apple Speech recognizer and the custom Qwen3-ASR implementation. Both operate offline, ensuring privacy and allowing usage without an internet connection.

## Prerequisites

*   A Mac with Xcode installed.
*   An iOS device (iPhone or iPad) with iOS 13.0 or later (for Apple Speech) or newer for MLX support (A-series chips recommended).
    *   **Note**: Offline transcription works best on physical devices.

## Setup Instructions

### 1. Basic Setup

1.  **Open Project**: Open `OfflineTranscriber.xcodeproj` (or create one following the steps below if starting from scratch).
2.  **Permissions**: Ensure `Info.plist` has `NSMicrophoneUsageDescription` and `NSSpeechRecognitionUsageDescription`.

### 2. Qwen3-ASR Setup (Required for MLX mode)

To use the Qwen3-ASR model, you must manually download the model weights and configure the `mlx-swift` dependency.

**Please refer to [AGENTS.md](AGENTS.md) for detailed instructions on:**
*   Adding the `mlx-swift` package dependency.
*   Downloading the required `model.safetensors`, `config.json`, and `tokenizer.json` files from Hugging Face.
*   Adding these files to your Xcode project bundle.

## Usage

1.  **Run the App**: Build and Run on a physical device.
2.  **Select Recognizer**: Use the segmented control at the top to choose between "Apple Speech" and "Qwen3-ASR (MLX)".
3.  **Start Recording**: Tap the button to start transcribing.
    *   **Apple Speech**: Streaming transcription (text appears as you speak).
    *   **Qwen3-ASR**: Offline batch transcription (text appears after you stop recording).

## Offline Capability

### Apple Speech
*   Uses `requiresOnDeviceRecognition = true`.
*   Requires Dictation to be enabled in iOS Settings.

### Qwen3-ASR
*   Runs completely locally using the `mlx-swift` framework.
*   Performance depends on the device hardware (Neural Engine).

## Troubleshooting

*   **"Model files missing"**: See `AGENTS.md` to ensure you downloaded and added the Qwen3 model files correctly.
*   **"Speech recognizer is not available"**: Check device language support for Apple Speech.
*   **Garbage Output (Qwen)**: The current implementation requires a robust Mel Spectrogram implementation (see code comments). The prototype may output silence or random tokens until the DSP layer is fully implemented.
