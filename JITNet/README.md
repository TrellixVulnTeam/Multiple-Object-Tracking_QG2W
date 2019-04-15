# Setting Up Google Compute Engine

Follow the instructions in [gce_machine_setup.md](gce_machine_setup.md) for prepping a machine with Tensorflow installed and mounting a disk. Pytorch can be installed afterwards.

# Running OSVOS Baseline

Sample usage: `python OSVOS-TensorFlow/osvos_seg.py --dataset_dir=/mnt/disks/tensorflow-disk/video_distillation/ --max_frames=18000 --training_stride=30 --sequence=giraffe1 --stats_path=/home/stevenzc3/osvos/giraffe1.npy -sequence_limit=2 --start_frame=5400 --height=480 --width=854`

The script requires the OSVOS parent model to be placed as instructed in the README of the OSVOS-TensorFlow repository.

# Running Flow Baseline

Sample usage: `python src/flow_seg.py --dataset_dir=/mnt/disks/tensorflow-disk/video_distillation/ --max_frames=100 --training_stride=10 --sequence=hockey1 --stats_path=/tmp/flow_distill.npy --start_frame=0 --height=1080 --width=1920`

The script requires PWCNet weights (download all files from https://drive.google.com/drive/folders/1gtGx_6MjUQC5lZpl6-Ia718Y_0pvcYou) to be placed in a folder parallel to the JITNet repository. For instance:

JITNet repo: `/home/stevenzc3/git/JITNet`
PWCNet weights: `/home/stevenzc3/git/pwcnet/pwcnet_lg/{files here}`

Important parameters:

* dataset_dir, sequence (same config as online_scene_distillation script)
* max_frames: number of frames to run from the video
* training_stride: parent network prediction is used for every frame `i % training_stride == 0`
* stats_path: optional, specify path and name for .npy file containing stats in a single element list
* start_frame: start frame within the video
* height, width: these must match the video and predictions

The stats npy file can be loaded in the following:

`stats = np.load('stats.npy')[0]`

`stats` is a dictionary containing the update_stats output for all non-teacher frames.

# Downloading Youtube Streams

Install the latest youtube-dl binary from here: http://rg3.github.io/youtube-dl/download.html

In order to download livestreams, ffmpeg needs to built from source with ssl support.

To do this, follow the instuctions for building from source for Ubuntu up to and before the final make command for ffmpeg: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

Install openssl sources with `sudo apt install libssl-dev`, and add the option `--enable-openssl` to the configure list,
then build.

The instructions put a line into ~/.profile: It doesn't seem that Ubuntu loads this by default, so instead, put it into ~/.bashrc.

To use youtube-dl, first run `youtube-dl -F <youtube URL>` to get a list of quality options. Choose the highest quality number
and download with `youtube-dl -f <option number> <youtube URL>`.

# Instructions for Running JITNet on Tegra
Hardware: NVIDIA Jetson TX2

* Flash JetPack 3.3 with a complete installation (check all optional installation modules)
* Install some necessary packages for TensorFlow install:
  ```sudo apt install python2.7-dev virtualenv```
* Create a python2 virtualenv:
  ```virtualenv --python=/usr/bin/python2.7 /home/nvidia/tensorflow2```
* Install Jetson TX2 Tensorflow for Python 2.7 at the following link: https://devtalk.nvidia.com/default/topic/1038957/jetson-tx2/tensorflow-for-jetson-tx2-/. We use Tensorflow 1.9 and install into the virtualenv as follows:
  ```/home/nvidia/tensorflow2/bin/pip install --extra-index-url=https://developer.download.nvidia.com/compute/redist/jp33 tensorflow-gpu```
* Activate the virtualenv: ```source /home/nvidia/tensorflow2/bin/activate```

Before running, set the Tegra to max-N mode for maximum performance:
```sudo nvpmodel -m 0; sudo ~/jetson_clocks.sh```

# Benchmarking JITNet
`python utils/time_models.py`
