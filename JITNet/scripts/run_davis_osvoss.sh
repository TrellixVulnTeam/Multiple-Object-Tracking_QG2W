for s in 'blackswan' 'bmx-trees' 'breakdance' 'camel' 'car-roundabout' 'car-shadow' 'cows' 'dance-twirl' 'dog' 'drift-chicane' 'drift-straight' 'goat' 'horsejump-high' 'kite-surf' 'libby' 'motocross-jump' 'paragliding-launch' 'parkour' 'scooter-black' 'soapbox'
do
    mkdir -p /home/stevenzc3/davis_jitnet_osvoss
    mkdir -p /home/stevenzc3/davis_jitnet_osvoss/${s}
    source scripts/online_train_davis_osvoss.sh 16 $s 1 /home/stevenzc3/davis_jitnet_osvoss/${s}
done
