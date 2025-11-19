# TODOs

Required:
- [ ] update the readme, agents, etc
- [ ] make the CLI flags shorter, no need to prefix them with `assistant`
- [ ] make the assistant reply even more succint
- [ ] set default model to nano
- [ ] the readme should be a readme not a dump
- [ ] add all cli commands for the project in docs/cli.md
- [ ] is there a way to automatically get the location
- [ ] --reset should not reset the api key
- [ ] make the whole app runnable without uv
- [ ] clean up /docs
- [ ] test if selecting via cli 5.1 super high reasoning actually selects it as it's very fast
- [ ] if i shut down the assistant the logs appear not to save, or at least the convo, even if i wait for it to finish
- [ ] add a way to reset the initial choices / go through them again
- [ ] fix hey x stop
- [ ] revisit what to keep in non -v logs
- [ ] revisit all logs
- [ ] Drop "hey wakeword" from user message
- [ ] Don't wait for sentence to finish, stream / transcribe in smaller chunks
- [ ] Measure real reply latency for each assistant model and update onboarding copy
- [ ] add loading dots to the logs when loading response from llm

Maybe:
- [ ] Stream the reply
- [ ] add a way to generate code coverage, write it in a gitignored file. it should have what entry points are missing unit tests
- [ ] wake word logs are too much in non -v
- [ ] explore if we can add more custom names for wake words
- [ ] Add speaker diarization
- [ ] Auto-detect microphone sample rate support and adjust capture/session config when defaults fail (avoid manual SAMPLE_RATE exports)
