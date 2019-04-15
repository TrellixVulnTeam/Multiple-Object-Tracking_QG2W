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

sequence_to_vis_name = { 'kabaddi1': 'Kabaddi',
                         'figure_skating1': 'Figure Skating',
                         'dogs2': 'Dog',
                         'biking1': 'Biking',
                         'samui_murphys1': 'Murphys',
                         'driving1': 'Driving'}

grouping = {
             'Overall' : sequence_to_vis_name.keys(),
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

print('\\arrayrulecolor{black!30}\\midrule\n \
        \multicolumn{7}{c}{Grouped}\\\\')

for group in grouping:
    column_headers = ['JITNet 0.8', \
                      'JITNet 0.8 max updates 4', \
                      'JITNet 0.8 max updates 16', \
                      'JITNet 0.8 lr 0.001', \
                      'JITNet 0.8 lr 0.1',\
                      'JITNet 0.8 min stride 4',\
                      'JITNet 0.8 min stride 16'
                      ]
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

        num_classes = len(class_groups) + 1
        f = open('tables_ablation/%s.csv'%(s))
        reader = csv.reader(f)
        column_vals = {}
        for row in reader:
            if row[0] in column_headers:
                mean_iou[row[0]] = mean_iou[row[0]] + np.mean([ float(v) for v in row[2:num_classes+1]])
                sample_fraction[row[0]] = sample_fraction[row[0]] + float(row[num_classes + 2])/300
                speed_up[row[0]] = speed_up[row[0]] + float(row[num_classes+1])

    num_seq = len(grouping[group])
    for h in column_headers:
        column_vals[h] = '%.1f(%.1f$\\times$, %.1f\\%%)'%(mean_iou[h]/num_seq,
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
