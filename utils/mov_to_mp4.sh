#!/bin/bash

input_dir="/Users/alessandrofolloni/Desktop/Unibo/PW ML4CV/videos/Converted"
output_dir="/Users/alessandrofolloni/Desktop/Unibo/PW ML4CV/videos/final_videos/"

mkdir -p "$output_dir"

for mov_file in "$input_dir"/*.mov; do
    filename=$(basename -- "$mov_file")
    filename_no_ext="${filename%.*}"
    mp4_file="$output_dir/$filename_no_ext.mp4"

    ffmpeg -i "$mov_file" -c:v libx264 -crf 23 -preset fast -c:a aac -strict experimental "$mp4_file"

    echo "Converted $filename to MP4."
done