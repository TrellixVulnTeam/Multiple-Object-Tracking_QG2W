for s in 'scooter-black' 'lucia' 'rollerblade' 'breakdance' 'camel' 'train' 'breakdance-flare' 'drift-straight' 'bear' 'dog-agility' 'bmx-trees' 'libby' 'cows' 'tennis' 'car-shadow' 'bus' 'surf' 'hike' 'swing' 'drift-chicane' 'dance-jump' 'horsejump-low' 'paragliding' 'dog' 'stroller' 'flamingo' 'elephant' 'rhino' 'mallard-water' 'horsejump-high' 'motocross-jump' 'goat' 'parkour' 'scooter-gray' 'motocross-bumps' 'mallard-fly' 'paragliding-launch' 'soapbox' 'dance-twirl' 'kite-surf' 'drift-turn' 'hockey' 'car-roundabout' 'blackswan' 'bmx-bumps' 'car-turn'
do
    mkdir -p /home/stevenzc3/davis_jitnet_stride
    mkdir -p /home/stevenzc3/davis_jitnet_stride/${s}
    source scripts/online_train_davis_stride.sh 16 $s 1 /home/stevenzc3/davis_jitnet_stride/${s}
done
