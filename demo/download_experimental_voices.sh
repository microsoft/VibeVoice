#!/usr/bin/env bash
set -e

echo "[INFO] Starting download of experimental voices..."

# Absolute path of the current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target directory relative to this script location
TARGET_DIR="$SCRIPT_DIR/voices/streaming_model/experimental_voices"

echo "[INFO] Script directory: $SCRIPT_DIR"
echo "[INFO] Target directory: $TARGET_DIR"

# Ensure the target directory exists
echo "[INFO] Creating target directory if needed..."
mkdir -p "$TARGET_DIR"

# List of archives and their URLs
FILES=(
  "experimental_voices_de.tar.gz|https://github.com/user-attachments/files/24035887/experimental_voices_de.tar.gz"
  "experimental_voices_fr.tar.gz|https://github.com/user-attachments/files/24035880/experimental_voices_fr.tar.gz"
  "experimental_voices_jp.tar.gz|https://github.com/user-attachments/files/24035882/experimental_voices_jp.tar.gz"
  "experimental_voices_kr.tar.gz|https://github.com/user-attachments/files/24035883/experimental_voices_kr.tar.gz"
  "experimental_voices_pl.tar.gz|https://github.com/user-attachments/files/24035885/experimental_voices_pl.tar.gz"
  "experimental_voices_pt.tar.gz|https://github.com/user-attachments/files/24035886/experimental_voices_pt.tar.gz"
  "experimental_voices_sp.tar.gz|https://github.com/user-attachments/files/24035884/experimental_voices_sp.tar.gz"
  "experimental_voices_en1.tar.gz|https://github.com/user-attachments/files/24189272/experimental_voices_en1.tar.gz"
  "experimental_voices_en2.tar.gz|https://github.com/user-attachments/files/24189273/experimental_voices_en2.tar.gz"
)

# Download, extract, and clean up each archive
for entry in "${FILES[@]}"; do
  IFS="|" read -r FNAME URL <<< "$entry"

  echo "[INFO] Downloading $FNAME ..."
  wget -O "$FNAME" "$URL"

  echo "[INFO] Extracting $FNAME ..."
  tar -xzvf "$FNAME" -C "$TARGET_DIR"

  echo "[INFO] Cleaning up $FNAME ..."
  rm -f "$FNAME"
done

echo "[SUCCESS] All experimental speakers installed successfully!"
echo "[SUCCESS] Speakers are located at:"
echo "          $TARGET_DIR"
