# TODOs

Required:
- [x] move test_recording.wav under tests/manual/
- [ ] update the readme, agents, etc
- [ ] Drop "hey wakeword" from user message
- [ ] Don't wait for sentence to finish, stream / transcribe in smaller chunks
- [ ] Measure real reply latency for each assistant model and update onboarding copy
- [ ] add timestamps to the logs so we see the bottlenecks

Maybe:
- [ ] Stream the reply
- [ ] wake word logs are too much in non -v
- [ ] explore if we can add more names for wake words
- [ ] Add speaker diarization
- [ ] Auto-detect microphone sample rate support and adjust capture/session config when defaults fail (avoid manual SAMPLE_RATE exports)
