for s in \
kabaddi1 samui_murphys1 samui_walking_street1 soccer1 softball1 southbeach1 squash1 streetcam1 streetcam2 table_tennis1 tennis1 tennis2 tennis3 toomer1 volleyball1 volleyball3 walking1
do
if [ ! -f /home/stevenzc3/osvos_static/${s}.npy ]; then
echo ${s}
python OSVOS-TensorFlow/osvos_seg.py --dataset_dir=/mnt/disks/tensorflow-disk/video_distillation/ --max_frames=3600 --training_stride=30 --sequence=${s} --stats_path=/home/stevenzc3/osvos_static/${s} -sequence_limit=2 --start_frame=5400 --height=480 --width=854
fi
done
