import sys, os
import csv
import numpy as np
sys.path.append('./utils')
sys.path.append('./datasets')
import video_distillation

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

grouping = {
            'Overall' : sequence_to_vis_name.keys(),
'Sports(Fixed)': ['badminton1', 'squash1', 'table_tennis1', 'softball1'],
             'Sports(Moving)': ['soccer1', 'tennis3', 'volleyball3', 'ice_hockey1', \
                               'kabaddi1', 'figure_skating1', 'drone2'],
             'Sports(Ego)': ['ice_hockey_ego_1', 'basketball_ego1', 'ego_dodgeball1', 'ego_soccer1', 'biking1'],
             'Animals' : ['elephant1', 'birds2', 'giraffe1', 'dogs2', 'horses1'],
             'Traffic': ['streetcam1', 'streetcam2', 'jackson_hole1', 'samui_murphys1', 'samui_walking_street1', 'toomer1'],
             'Driving/Walking' : ['driving1', 'walking1'],
             'Fixed' : ['badminton1', 'squash1', 'table_tennis1', 'softball1', 'birds2', \
                        'streetcam1', 'streetcam2', 'jackson_hole1', 'samui_murphys1', \
                        'samui_walking_street1', 'toomer1'],
           }

class_acronyms = { 'person': 'P',
                   'car': 'C',
                   'bus': 'C',
                   'truck': 'C',
                   'bicycle': 'B',
                   'motorcycle': 'B',
                   'bird': 'B',
                   'elephant': 'E',
                   'horse': 'H',
                   'dog': 'D',
                   'giraffe': 'G' }

for group in grouping:
    column_headers = ['OSVOS', 'OSVOS static', 'JITNet 0.8']
    mean_iou = {}
    sample_fraction = {}
    speed_up = {}

    for c in column_headers:
        mean_iou[c] = 0
        sample_fraction[c] = 0
        speed_up[c] = 0

    for s in grouping[group]:
        if s not in sequence_to_vis_name:
            continue
        class_groups = video_distillation.sequence_to_class_groups_stable[s]
        classes = [ c for g in class_groups for c in g ]

        #num_classes = len(class_groups) + 1
        num_classes = 2
        f = open('tables_osvos/%s.csv'%(s))
        reader = csv.reader(f)
        column_vals = {}
        for row in reader:
            if row[0] in column_headers:
                mean_iou[row[0]] = mean_iou[row[0]] + np.mean([ float(v) for v in row[2:num_classes+1]])
                sample_fraction[row[0]] = sample_fraction[row[0]] + float(row[len(class_groups) + 3])/36
                speed_up[row[0]] = speed_up[row[0]] + float(row[len(class_groups)+2])

    num_seq = len(grouping[group])
    for h in column_headers:
        if h == 'OSVOS static' or h == 'OSVOS':
            column_vals[h] = '%.1f'%(mean_iou[h]/num_seq)
        elif h == 'JITNet 0.8':
            column_vals[h] = '%.1f($\\times$%.1f, %.1f\\%%)'%(mean_iou[h]/num_seq,
                                                              speed_up[h]/num_seq,
                                                              sample_fraction[h]/num_seq)
        else:
            column_vals[h] = '%.1f($\\times$%.1f, %.1f\\%%)'%(mean_iou[h]/num_seq,
                                                              speed_up[h]/num_seq,
                                                              sample_fraction[h]/num_seq)

    print('\\arrayrulecolor{black!30}\midrule')
    row = ''
    for h in column_headers:
        if h in column_vals:
            row = row + '& ' + column_vals[h]
        else:
            row = row + '& ' + '-'
    print('{%s}%s\\\\'%(group, row))

for s in flat_list:
    if s not in sequence_to_vis_name:
        continue
    class_groups = video_distillation.sequence_to_class_groups_stable[s]
    #num_classes = len(class_groups) + 1
    num_classes = 2
    f = open('tables_osvos/%s.csv'%(s))
    reader = csv.reader(f)
    column_headers = ['OSVOS', 'OSVOS static', 'JITNet 0.8']
    column_vals = {}
    for row in reader:
        if row[0] != '':
            mean_iou = np.mean([ float(v) for v in row[2:num_classes+1]])
            speed_up = float(row[len(class_groups) + 2])
            sample_fraction = float(row[len(class_groups) + 3])/36
            if row[0] == 'JITNet 0.8':
                column_vals[row[0]] = '%.1f($\\times$%.1f, %.1f\\%%)'%(mean_iou, speed_up, sample_fraction)
            else:
                column_vals[row[0]] = '%.1f'%(mean_iou)

    print('\\arrayrulecolor{black!30}\midrule')
    row = ''
    for h in column_headers:
        if h in column_vals:
            row = row + '& ' + column_vals[h]
        else:
            row = row + '& ' + '-'
    print('{%s}%s\\\\'%(sequence_to_vis_name[s], row))
