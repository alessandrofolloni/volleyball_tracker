#!/bin/bash

# Base folder path
BASE_FOLDER_PATH="$HOME/Desktop/Unibo/PW ML4CV/videos"

# Subfolders for converted files
CONVERTED_FOLDER="Converted"
FINAL_FOLDER="final_videos"

# Full paths for output directories
CONVERTED_PATH="$BASE_FOLDER_PATH/$CONVERTED_FOLDER"
FINAL_PATH="$BASE_FOLDER_PATH/$FINAL_FOLDER"

# Create output directories if they don't exist
mkdir -p "$CONVERTED_PATH"
mkdir -p "$FINAL_PATH"

# Function to convert .m4v files to .mp4 and save in the Converted folder
convert_m4v_to_mp4() {
  local input_folder="$1"
  find "$input_folder" -type f -name "*.m4v" | while read -r file; do
    # Determine relative path and output directory
    relative_path="${file#$BASE_FOLDER_PATH/}"
    output_dir="$CONVERTED_PATH/$(dirname "$relative_path")"
    mkdir -p "$output_dir"

    # Convert to MP4 using ffmpeg
    ffmpeg -i "$file" -codec copy "$output_dir/${file%.m4v}.mp4"
    echo "Converted $(basename "$file") to MP4 in $output_dir."
  done
}

# Function to convert .mov files to .mp4 and save in the Final folder
convert_mov_to_mp4() {
  local input_folder="$1"
  find "$input_folder" -type f -name "*.mov" | while read -r mov_file; do
    # Determine output file name
    filename=$(basename -- "$mov_file")
    filename_no_ext="${filename%.*}"
    mp4_file="$FINAL_PATH/$filename_no_ext.mp4"

    # Convert to MP4 using ffmpeg
    ffmpeg -i "$mov_file" -c:v libx264 -crf 23 -preset fast -c:a aac -strict experimental "$mp4_file"
    echo "Converted $filename to MP4 in $FINAL_PATH."
  done
}

# Convert all .m4v files in the base folder to .mp4
convert_m4v_to_mp4 "$BASE_FOLDER_PATH"

# Convert all .mov files in the converted folder to .mp4
convert_mov_to_mp4 "$CONVERTED_PATH"

echo "All conversions completed!"


