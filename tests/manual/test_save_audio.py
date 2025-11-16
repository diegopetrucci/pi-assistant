"""
Test script that saves captured audio to a WAV file so you can verify it's working
"""

import asyncio
import sys
import wave

from pi_transcription.audio import AudioCapture
from pi_transcription.config import CHANNELS, SAMPLE_RATE


async def test_and_save_audio():
    """Capture audio and save to WAV file for verification"""
    print("\n=== Audio Capture & Save Test ===\n")

    # Create audio capture instance
    capture = AudioCapture()

    # Get current event loop
    loop = asyncio.get_running_loop()

    # Start audio stream
    capture.start_stream(loop)

    print("Recording for 5 seconds...")
    print('Speak into your microphone: "Testing, one, two, three"\n')

    audio_chunks = []
    chunk_count = 0

    try:
        # Capture for 5 seconds
        start_time = loop.time()
        while loop.time() - start_time < 5.0:
            try:
                audio_data = await asyncio.wait_for(capture.get_audio_chunk(), timeout=1.0)
                audio_chunks.append(audio_data)
                chunk_count += 1

                # Show progress
                if chunk_count % 20 == 0:
                    elapsed = loop.time() - start_time
                    print(f"Recording... {elapsed:.1f}s")

            except asyncio.TimeoutError:
                print("Error: No audio data received")
                break

    except KeyboardInterrupt:
        print("\nRecording interrupted")

    finally:
        capture.stop_stream()

    # Save to WAV file
    if audio_chunks:
        output_file = "test_recording.wav"
        print(f"\nSaving {chunk_count} chunks to {output_file}...")

        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(b"".join(audio_chunks))

        print("✅ Saved successfully!")
        print("\nTo verify the recording, play it back:")
        print(f"  afplay {output_file}  (macOS)")
        print(f"  aplay {output_file}   (Linux/Raspberry Pi)")
        print("\nIf you hear your voice, the audio capture is working correctly!")
    else:
        print("❌ No audio captured")


def main():
    try:
        asyncio.run(test_and_save_audio())
    except KeyboardInterrupt:
        print("\nTest cancelled")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
