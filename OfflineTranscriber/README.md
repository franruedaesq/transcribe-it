# Offline iOS Transcriber

This is a SwiftUI application that demonstrates how to perform offline speech transcription on iOS using the native `Speech` framework (`SFSpeechRecognizer`).

## Overview

The app uses `SFSpeechRecognizer` with the `requiresOnDeviceRecognition` flag set to `true`. This ensures that the transcription happens locally on the device and does not send audio data to Apple's servers. This is crucial for privacy and offline usage.

## Prerequisites

*   A Mac with Xcode installed.
*   An iOS device (iPhone or iPad) with iOS 13.0 or later.
    *   **Note**: Offline transcription works best on physical devices. The Simulator may not support microphone input or offline speech models correctly.

## Setup Instructions

1.  **Create a New Xcode Project**:
    *   Open Xcode.
    *   Select "Create a new Xcode project".
    *   Choose "App" under the iOS tab.
    *   Click "Next".
    *   Product Name: `OfflineTranscriber` (or any name you prefer).
    *   Interface: `SwiftUI`.
    *   Language: `Swift`.
    *   Click "Next" and choose a location to save the project.

2.  **Add Source Files**:
    *   In the Xcode project navigator (left sidebar), verify you have `OfflineTranscriberApp.swift` and `ContentView.swift`.
    *   Copy the content of `OfflineTranscriberApp.swift` from this repository into the file in Xcode.
    *   Copy the content of `ContentView.swift` from this repository into the file in Xcode.
    *   Create a new Swift file named `SpeechRecognizer.swift` in Xcode (File > New > File > Swift File) and copy the content from this repository into it.

3.  **Configure Info.plist**:
    *   The app requires permission to use the Microphone and Speech Recognition.
    *   Click on the root project node in the Project Navigator.
    *   Select the "Info" tab.
    *   Add a new key `Privacy - Microphone Usage Description` (`NSMicrophoneUsageDescription`) with a value like "This app requires microphone access to record audio for transcription."
    *   Add a new key `Privacy - Speech Recognition Usage Description` (`NSSpeechRecognitionUsageDescription`) with a value like "This app requires speech recognition access to transcribe your speech."
    *   **Alternatively**: You can verify the `Info.plist` file content provided in this repository matches your project configuration.

4.  **Run the App**:
    *   Connect your iOS device to your Mac.
    *   Select your device as the run destination in Xcode.
    *   Build and Run (Cmd + R).
    *   On the device, accept the permissions for Microphone and Speech Recognition when prompted.
    *   Tap "Start Recording" and speak. The text should appear on the screen.

## Offline Capability

*   The code sets `request.requiresOnDeviceRecognition = true`.
*   If the device does not have the necessary language model downloaded for offline use, the recognition might fail.
*   To ensure offline support works, go to **Settings > General > Keyboard > Dictation** on your iOS device and ensure Dictation is enabled. iOS automatically downloads the necessary models for supported languages.
*   If `requiresOnDeviceRecognition` fails, it usually means the on-device model is not available for the current locale.

## Troubleshooting

*   **"Speech recognizer is not available"**: Check if your device supports speech recognition and if the language is supported.
*   **"Permission denied..."**: Go to Settings > Privacy on your device and check if the app has access to Microphone and Speech Recognition.
*   **No transcription appears**: Ensure you are speaking clearly and the volume is sufficient. Check the debug console in Xcode for any errors.
