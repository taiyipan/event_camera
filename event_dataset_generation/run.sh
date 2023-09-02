#!/bin/bash
# define paths 
raw=/home/taiyi/scratch2/NYU-EventVPR
event=/home/taiyi/scratch/event_rendered/10fps
output=/home/taiyi/scratch/NYU-EventVPR-Event
# purge previous output directory 
rm -r $output 
echo "Purged previous dataset directory"
# ffmpeg frame generation 
bash event_to_img.sh
# process and create event dataset 
python /home/taiyi/scratch/event_dataset_generation/main.py \
       --rdir $raw \
       --dir $output \
       --edir $event \
       --tolerance 25 \
       --sobel 100 \
       --reduce 1.0 \
       --sample 0.1 \
       --framerate 1 \
       --synched 0