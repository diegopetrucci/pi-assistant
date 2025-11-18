# TODOs

Required:
- [ ] update the readme, agents, etc
- [ ] Drop "hey wakeword" from user message
- [ ] Don't wait for sentence to finish, stream / transcribe in smaller chunks
- [ ] Measure real reply latency for each assistant model and update onboarding copy
- [ ] add loading dots to the logs when loading response from llm

Maybe:
- [ ] Stream the reply
- [ ] wake word logs are too much in non -v
- [ ] explore if we can add more custom names for wake words
- [ ] Add speaker diarization
- [ ] Auto-detect microphone sample rate support and adjust capture/session config when defaults fail (avoid manual SAMPLE_RATE exports)
