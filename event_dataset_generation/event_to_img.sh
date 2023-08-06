#!/bin/bash

# parent directory 
dir=/mnt/scratch
# event render video paths 
files="$dir/data_processed/render/10fps/data_*/event_*.avi"
# initialize output frame directory 
output=$dir/event_rendered/10fps
rm -r $output
mkdir $output 

# iterate over all video files 
for file in $files
do 
    # extract extract and remove extension
    filename=$(basename -- "$file")
    filename=${filename%.*}
    echo "Processing $file"
    
    # create output frame directory for targeted event video 
    mkdir $output/$filename/
    echo "Output directory $output/$filename"

    # sample event video into frames 
    ffmpeg -hwaccel cuda -i $file $output/$filename/%07d.jpg 
done 