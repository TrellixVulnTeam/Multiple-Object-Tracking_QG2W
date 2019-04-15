for s in \
badminton1 basketball_ego1 biking1 birds2 dogs2 driving1 drone2 ego_dodgeball1 ego_soccer1 elephant1 figure_skating1 giraffe1 hockey1 horses1 ice_hockey1 ice_hockey_ego_1 jackson_hole1 jackson_hole2 kabaddi1 samui_murphys1 samui_walking_street1 soccer1 softball1 southbeach1 squash1 streetcam1 streetcam2 table_tennis1 tennis1 tennis2 tennis3 toomer1 volleyball1 volleyball3 walking1
do
if [ ! -f /home/stevenzc3/osvos/${s}.npy ]; then
echo ${s}
python OSVOS-TensorFlow/osvos_seg.py --dataset_dir=/mnt/disks/tensorflow-disk/video_distillation/ --max_frames=3600 --training_stride=30 --sequence=${s} --stats_path=/home/stevenzc3/osvos/${s} -sequence_limit=2 --start_frame=5400 --height=480 --width=854
fi
done
