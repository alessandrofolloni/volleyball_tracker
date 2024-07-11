#!/bin/bash

FOLDER_PATH="$HOME/Desktop/Unibo/PW ML4CV/videos"
# Nome della sottocartella per i file MP4 convertiti
CONVERTED_FOLDER="Converted"

mkdir -p "$FOLDER_PATH/$CONVERTED_FOLDER"

convert_files() {
  local folder_path="$1"
  # shellcheck disable=SC2095
  find "$folder_path" -type f -name "*.m4v" | while read -r file; do
    # Crea la stessa struttura di directory all'interno della cartella Converted
    relative_path="${file#$FOLDER_PATH/}"
    output_dir="$FOLDER_PATH/$CONVERTED_FOLDER/$(dirname "$relative_path")"
    mkdir -p "$output_dir"
    # Converti il file
    ffmpeg -i "$file" -codec copy "$output_dir/${file%.m4v}.mp4"
  done
}

convert_files "$FOLDER_PATH"