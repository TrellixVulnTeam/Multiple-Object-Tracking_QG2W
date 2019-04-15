import argparse
import subprocess
import os
import signal
import sys
from time import sleep

parser = argparse.ArgumentParser(description='Convert video to frames')
parser.add_argument('--video_path', type=str, help='Path to the video file')
parser.add_argument('--output_dir', type=str, help='Path to the directory to store extracted frames')
parser.add_argument('--fps', type=str, help='Rate at which frames are extracted')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

video_name = '.'.join(args.video_path.split('/')[-1].split('.')[:-1])
arg_string = '-i %s -vf fps=%s %s/%s'%(args.video_path, args.fps, args.output_dir, video_name)
arg_string = arg_string + '%05d.png'

cmd_string = ' '.join(['ffmpeg', arg_string])
process = subprocess.Popen(cmd_string, shell=True)
process.wait()
