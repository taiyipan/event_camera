#!/bin/bash

# event render video paths 
files="/mnt/scratch2/NYU-EventVPR_rendered_10fps/data_*/event_*.avi"
# initialize output frame directory 
output=/mnt/scratch/event_rendered/10fps
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
    ffmpeg -hwaccel cuda -i $file -vf fps=1 $output/$filename/%07d.jpg 
done 