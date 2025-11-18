# Automatic Location Discovery Options

1. **Public IP Geolocation**
   - Query services such as ipinfo, MaxMind GeoLite2, or ip-api to resolve the Pi's outward-facing IP to a lat/long + city.
   - Accuracy: ~5–50 km depending on ISP/NAT; good enough for city-level defaults.
   - Implementation: hit API during startup, cache for ~24h, fall back to manual input if response is vague.
   - Privacy: discloses IP to third-party service; mitigate with provider that supports self-hosted DB or on-box GeoLite.

2. **Wi-Fi Access Point Fingerprinting**
   - Scan for nearby SSIDs/BSSIDs via `iw`/`nmcli`, submit hashed MACs to Mozilla Location Service or Google Geolocation API for triangulation.
   - Accuracy: tens of meters in urban areas with dense AP catalog coverage.
   - Implementation: requires Wi-Fi interface in monitor mode; throttle scans to avoid audio glitches.
   - Privacy: reveals surrounding network IDs; consider opting into MLS self-hosted mirror.

3. **GPS/GNSS Module**
   - Attach a USB or UART GNSS receiver (u-blox, Pi HAT) and stream NMEA sentences for precise coordinates.
   - Accuracy: 3–5 m with clear sky; degraded indoors unless paired with external antenna.
   - Implementation: read from `/dev/ttyAMA0` (UART) or `/dev/ttyACM0` (USB) using `gpsd` or custom parser.
   - Privacy: fully local, but hardware cost and need for antenna placement.

4. **Cellular Modem Triangulation**
   - Use LTE HAT or USB modem capable of reporting serving cell IDs and signal strength; query OpenCellID or MLS cell database.
   - Accuracy: ~100–1000 m depending on tower density.
   - Implementation: requires SIM with data plan; parse AT commands (e.g., `AT+CREG?`, `AT+QENG`), send to API for lookup.
   - Privacy: shares cell IDs with lookup service; consider local copy of OpenCellID database.

5. **Bluetooth Beacon Mapping**
   - Scan for BLE beacons (iBeacon, Eddystone) and match UUID/MAC to known locations (retail, offices, custom tags).
   - Accuracy: room-level if beacons are pre-mapped.
   - Implementation: `bluetoothctl`/BlueZ for scanning plus internal mapping table or vendor API (e.g., Estimote Cloud).
   - Privacy: local if we own the beacon map; third-party if using vendor cloud.

6. **Host OS Location Service Bridge**
   - On desktop deployments, read coordinates from system services (macOS CoreLocation, Linux GeoClue2, Windows GeoLocation API) that already blend Wi-Fi, IP, and sensors.
   - Accuracy: same as OS-level heuristics (usually <100 m in cities).
   - Implementation: add optional client that speaks DBus to GeoClue or uses `locationd` socket on macOS when running the Pi assistant off-board.
   - Privacy: inherits OS permissions prompts; keeps third-party calls centralized.
