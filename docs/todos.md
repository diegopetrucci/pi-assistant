# TODOs

Required:
- [ ] Drop "hey wakeword" from user message
- [ ] force the language to be english
- [ ] Don't wait for sentence to finish, stream / transcribe in smaller chunks
- [ ] let user pick name
- [ ] Measure real reply latency for each assistant model and update onboarding copy
- [ ] add timestamps to the logs so we see the bottlenecks

Maybe:
- [ ] Option to reduce debug logs
- [ ] add pylance
- [ ] Add speaker diarization
- [ ] Auto-detect microphone sample rate support and adjust capture/session config when defaults fail (avoid manual SAMPLE_RATE exports)
