import os, sys, glob
import argparse
sys.path.append('./scripts')
from ssh_runner import Runner

parser = argparse.ArgumentParser(description='Convert videos to frames')
parser.add_argument('--video_dir', type=str, help='Path to the video file directory')
parser.add_argument('--output_dir', type=str, help='Path to the directory to store extracted frames')
parser.add_argument('--fps', type=str, help='Rate at which frames are extracted')
parser.add_argument('--regex', type=str, help='File pattern to match videos', default='*.mp4')


resources = [
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
              ('rmullapu@pismo.pdl.local.cmu.edu', 0),
            ]

args = parser.parse_args()

vid_path_regex = os.path.join(args.video_dir, args.regex)

vid_paths = glob.glob(vid_path_regex)

tasks = []

for v in vid_paths:
    task_args = {}
    task_args['--video_path'] = v
    video_name = '.'.join(v.split('/')[-1].split('.')[:-1])
    task_args['--output_dir'] = os.path.join(args.output_dir, video_name)
    task_args['--fps'] = args.fps

    tasks.append(task_args)

def task_fn(**kwargs):
    arg_string = '--video_path=%s --fps=%s --output_dir=%s' % (kwargs['--video_path'],
                 kwargs['--fps'],
                 kwargs['--output_dir'])

    source_bash = 'source ~/.bash_profile; '
    change_dir = 'cd /n/pana/scratch/ravi/JITNet; '
    cmd = 'python scripts/video_to_frames.py ' + arg_string

    full_cmd = source_bash + change_dir + cmd
    return full_cmd

Runner(resources, tasks, task_fn)
