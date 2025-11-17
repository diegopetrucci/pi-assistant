#!/usr/bin/env bash
set -euo pipefail
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script requires apt-get (Debian/Raspberry Pi OS)."
  exit 1
fi
sudo apt-get update
sudo apt-get install -y \
  libportaudio2 \
  libportaudiocpp0 \
  portaudio19-dev
