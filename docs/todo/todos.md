# TODOs

Required:
- [ ] remove start.py
- [ ] move test_recording.wav to an assets folder
- [ ] update the readme, agents, etc
- [ ] Drop "hey wakeword" from user message
- [ ] Don't wait for sentence to finish, stream / transcribe in smaller chunks
- [ ] let user pick name
- [ ] Measure real reply latency for each assistant model and update onboarding copy
- [ ] add timestamps to the logs so we see the bottlenecks

Maybe:
- [ ] add pylance
- [ ] add audio feedback to let the user know it heard them
- [ ] Stream the reply
- [ ] Add speaker diarization
- [ ] Auto-detect microphone sample rate support and adjust capture/session config when defaults fail (avoid manual SAMPLE_RATE exports)
