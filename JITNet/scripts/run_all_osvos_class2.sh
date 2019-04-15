for s in \
driving1 \
basketball_ego1 \
jackson_hole1 \
jackson_hole2 \
giraffe1 \
toomer1 \
drone1 \
biking1 \
samui_walking_street1 \
horses1 \
walking1 \
samui_murphys1 \
dogs2 \
biking2 \
southbeach1 \
streetcam1 \
streetcam2
do
if [ ! -f /home/stevenzc3/osvos_second_class2/${s}.npy ]; then
echo ${s}
python OSVOS-TensorFlow/osvos_seg.py --dataset_dir=/mnt/disks/tensorflow-disk/video_distillation/ --max_frames=3600 --training_stride=30 --sequence=${s} --stats_path=/home/stevenzc3/osvos_second_class2/${s} --class_index=2 --sequence_limit=2 --start_frame=15000 --height=480 --width=854
fi
done
