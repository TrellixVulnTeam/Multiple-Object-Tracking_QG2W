for s in hockey1 tennis1 tennis2 tennis3 ice_hockey_ego_1 basketball_ego1 volleyball1 \
         volleyball2 volleyball3 ego_soccer1 soccer1 dodgeball1 ego_dodgeball1 driving1 \
         walking1 jackson_hole1 toomer1 jackson_hole2 park1 samui_walking_street1 \
         badminton1 squash1 drone2 biking1 giraffe1 birds2 samui_murphys1 dogs2 elephant1
do
    source scripts/online_distillation.sh ${s} 25 15000 0.5 $1
done
