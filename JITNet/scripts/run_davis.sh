for s in 'scooter-black' 'lucia' 'rollerblade' 'breakdance' 'camel' 'train' 'breakdance-flare' 'drift-straight' 'bear' 'dog-agility' 'bmx-trees' 'libby' 'cows' 'tennis' 'car-shadow' 'bus' 'surf' 'hike' 'swing' 'drift-chicane' 'dance-jump' 'horsejump-low' 'paragliding' 'dog' 'stroller' 'flamingo' 'elephant' 'rhino' 'mallard-water' 'horsejump-high' 'motocross-jump' 'goat' 'parkour' 'scooter-gray' 'motocross-bumps' 'mallard-fly' 'paragliding-launch' 'soapbox' 'dance-twirl' 'kite-surf' 'drift-turn' 'hockey' 'car-roundabout' 'blackswan' 'bmx-bumps' 'car-turn'
do
    mkdir -p /home/stevenzc3/davis_jitnet/${s}
    source scripts/online_train_davis.sh 50 $s 0 /home/stevenzc3/davis_jitnet/${s}
    ffmpeg -i /home/stevenzc3/davis_jitnet_videos/$s.avi /home/stevenzc3/davis_jitnet_videos/$s.mp4
done
