import sys, os
sys.path.append('./utils')
sys.path.append('./datasets')
import video_distillation
from subprocess import call

group_list = [['badminton1', 'squash1', 'table_tennis1',
                 'softball1', 'hockey1', 'soccer1', 'tennis1',
                 'tennis2', 'tennis3', 'volleyball1', 'volleyball3',
                 'ice_hockey1', 'kabaddi1', 'figure_skating1',
                 'drone2'],
              ['elephant1', 'birds2', 'giraffe1', 'dogs2', 'horses1'],
              ['ice_hockey_ego_1', 'basketball_ego1', 'ego_dodgeball1',
               'ego_soccer1', 'biking1'],
              ['streetcam1', 'streetcam2', 'jackson_hole1', 'jackson_hole2',
               'samui_murphys1', 'samui_walking_street1', 'toomer1',
               'southbeach1'],
              ['driving1', 'walking1']]
flat_list = [ v for g in group_list for v in g ]

sequence_to_vis_name = { 'badminton1' : 'Badminton',
                         'squash1' : 'Squash',
                         'table_tennis1' : 'Table Tennis',
                         'softball1' : 'Softball',
                         'hockey1' : 'Hockey',
                         'soccer1' : 'Soccer',
                         'tennis3' : 'Tennis',
                         'volleyball3' : 'Volleyball',
                         'ice_hockey1' : 'Ice Hockey',
                         'kabaddi1': 'Kabaddi',
                         'figure_skating1': 'Figure Skating',
                         'drone2': 'Drone',
                         'elephant1': 'Elephant',
                         'birds2': 'Birds',
                         'giraffe1': 'Giraffe',
                         'dogs2': 'Dog',
                         'horses1': 'Horse',
                         'ice_hockey_ego_1' : 'Ego Ice Hockey',
                         'basketball_ego1' : 'Ego Basketball',
                         'ego_dodgeball1' : 'Ego Dodgeball',
                         'ego_soccer1': 'Ego Soccer',
                         'biking1': 'Biking',
                         'streetcam1': 'Streetcam1',
                         'streetcam2': 'Streetcam2',
                         'jackson_hole1': 'Jackson Hole',
                         'samui_murphys1': 'Murphys',
                         'samui_walking_street1': 'Samui Street',
                         'toomer1': 'Toomer',
                         'driving1': 'Driving',
                         'walking1': 'Walking'
}

video_dir = '/n/pana/scratch/ravi/video_distillation_final'
out_dir = '/n/pana/scratch/ravi/video_distillation_supp'
for s in flat_list:
    if s not in sequence_to_vis_name:
        continue
    video_path = os.path.join(video_dir, s + '_stride_8_frame_30000_thresh_0.9_0.01.mp4')
    video_out_path = os.path.join(out_dir, sequence_to_vis_name[s].replace(' ', '') + '.mp4')
    video_out_path_fast = os.path.join(out_dir, sequence_to_vis_name[s].replace(' ', '') + 'Fast.mp4')
    os.system('ffmpeg -i %s -ss 00:06:00 -t 00:01:00 -filter:v "crop=2560:720:1280:0" %s'%(video_path, video_out_path))
    os.system('ffmpeg -i %s -filter:v "setpts=0.25*PTS" %s'%(video_out_path, video_out_path_fast))

#ffmpeg -i test.mp4 -filter:v "setpts=0.25*PTS" test_fast.mp4
#ffmpeg -i video_distillation_final/softball1_stride_8_frame_30000_thresh_0.9_0.01.mp4 -ss 00:06:00 -t 00:01:00 -filter:v "crop=2560:720:1280:0" test.mp4
