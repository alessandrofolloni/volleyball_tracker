#!/bin/bash

# Specify the full path to your folder
FOLDER_PATH="$HOME/Desktop/Unibo/PW ML4CV/videos"
# Name of the subfolder for converted MP4 files
CONVERTED_FOLDER="Converted"

# Create the Converted folder if it doesn't exist
mkdir -p "$FOLDER_PATH/$CONVERTED_FOLDER"

# Navigate to the folder containing the original files
cd "$FOLDER_PATH" || exit 1 # Exit if the folder doesn't exist

# Convert each M4V file to MP4 and save in the Converted folder
for file in *.m4v; do
  ffmpeg -i "$file" -codec copy "$CONVERTED_FOLDER/${file%.m4v}.mp4"
done


