import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import seaborn as sns
import cv2

import sys, os, glob
sys.path.append('./datasets')
import video_distillation

def make_data_frame(path_to_stats, classes, fps, sec_agg):
    stats_dict = np.load(path_to_stats)[0]
    frames = sorted(stats_dict.keys())

    entropy = []
    correct = []
    total = []
    accuracy = []
    time = []
    class_iou = {}
    class_acc = {}
    updates = []
    samples = []
    for c in classes:
        class_iou[c] = []
        class_acc[c] = []

    interval = fps * sec_agg
    for s in range(0, int(len(frames)/interval)):
        sec_entropy = 0
        sec_correct = 0
        sec_total = 0
        sec_class_tp = {}
        sec_class_fp = {}
        sec_class_fn = {}
        sec_class_correct = {}
        sec_class_total = {}
        sec_updates = 0
        sec_samples = 0
        for c in classes:
            sec_class_fp[c] = 0
            sec_class_tp[c] = 0
            sec_class_fn[c] = 0
            sec_class_correct[c] = 0
            sec_class_total[c] = 0

        for fidx in range(s*interval, (s+1)*interval):
            sec_entropy = sec_entropy + stats_dict[fidx]['average_entropy']
            sec_correct = sum(stats_dict[fidx]['correct']) + sec_correct
            sec_total = sum(stats_dict[fidx]['total']) + sec_total
            sec_updates = stats_dict[fidx]['num_updates'] + sec_updates
            sec_samples = stats_dict[fidx]['ran_teacher'] + sec_samples
            for c in range(len(classes)):
                sec_class_tp[classes[c]] = stats_dict[fidx]['tp'][c] + sec_class_tp[classes[c]]
                sec_class_fp[classes[c]] = stats_dict[fidx]['fp'][c] + sec_class_fp[classes[c]]
                sec_class_fn[classes[c]] = stats_dict[fidx]['fn'][c] + sec_class_fn[classes[c]]
                sec_class_correct[classes[c]] = stats_dict[fidx]['correct'][c] + sec_class_correct[classes[c]]
                sec_class_total[classes[c]] = stats_dict[fidx]['total'][c] + sec_class_total[classes[c]]

        eps = 32 * 32 * fps * sec_agg
        for c in classes:
            iou = (sec_class_tp[c] + eps)/(sec_class_tp[c] + sec_class_fp[c] + sec_class_fn[c] + eps)
            acc = (sec_class_correct[c] + eps)/(sec_class_total[c] + eps)
            class_iou[c].append(iou)
            class_acc[c].append(acc)
        correct.append(sec_correct)
        total.append(sec_total)
        entropy.append(sec_entropy/interval)
        accuracy.append(float(sec_correct)/sec_total)
        time.append((s * interval)/fps)
        updates.append(sec_updates)
        samples.append(sec_samples)

    d = {'Average_entropy': entropy,
         'Correct': correct,
         'Total': total,
         'Accuracy': accuracy,
         'Time(s)': time,
         'Updates': updates,
         'Samples': samples
         }

    iou_lists = []
    num_classes = len(class_iou.keys())
    for c in classes:
        d[c + '(IoU)'] = class_iou[c]
        d[c + '(Acc)'] = class_acc[c]
        iou_lists.append(class_iou[c])

    mean_iou = [ float(sum(item))/num_classes for item in zip(*iou_lists) ]
    d['Mean(IoU)'] = mean_iou
    df = pd.DataFrame(d)

    return df

c1 = (95.0/225, 77.0/255, 49.0/255)
c2 = (239.0/255, 111.0/255, 108.0/255)
c3 = (50.0/255, 71.0/255, 52.0/255)
c4 = (38.0/255, 30.0/255, 67.0/255)

dataset_dir =  '/n/scanner/ravi/video_distillation/'

results_dirs = ['/n/pana/scratch/ravi/video_distillation_final/',
                '/n/pana/scratch/ravi/video_distillation_overfit/']

suffixes = ['_stride_8_frame_30000_thresh_0.8_lr_0.01',
            '_stride_8_frame_30000_overfit']

names = ['JITNet 0.8', 'Overfit']

colors = [ c1, c2, c3 ]

time_interval = 30
fps = 25
num_images = 6

video_lists = video_distillation.get_sequence_to_video_list(dataset_dir,
                        dataset_dir, video_distillation.video_sequences_stable)

def get_frames(sequence, frame_list):
    frames = {}

    videos_path = os.path.join(dataset_dir, sequence)
    vid_list = video_lists[sequence][:2]
    vid_list = [ os.path.join(videos_path, v[0]) for v in vid_list]

    segment = 0
    curr_offset = 0

    v = cv2.VideoCapture(vid_list[segment])
    curr_limit = v.get(cv2.CAP_PROP_FRAME_COUNT)
    for fidx in sorted(frame_list):
        if fidx > curr_limit:
            segment = segment + 1
            v = cv2.VideoCapture(vid_list[segment])
            curr_offset = curr_limit
            curr_limit = curr_limit + v.get(cv2.CAP_PROP_FRAME_COUNT)

        v.set(cv2.CAP_PROP_POS_FRAMES, fidx - curr_offset)
        ret, image = v.read()
        frames[fidx] = image[...,::-1]

    return frames

for s in video_distillation.video_sequences_stable:
    print(s)
    class_names = video_distillation.sequence_to_class_groups_stable[s]
    classes = ['background'] + [ '_'.join(g) for g in class_names ]

    #fig, axes = plt.subplots(num_plots, 1, sharex=True)
    fig = plt.figure(figsize=(10 * 1.7777, 4))
    gs = GridSpec(2, 5, wspace=0.025, hspace=0.025)
    ax0 = plt.subplot(gs[0, :2])
    ax1 = plt.subplot(gs[1, :2])

    ax0.set_ylim(0.5, 1.1)

    dfs = {}
    legends = []

    for results_dir, suffix, color, name in  zip(results_dirs, suffixes, colors, names):
        results_path = os.path.join(results_dir, s + suffix + '.npy')
        df = make_data_frame(results_path, classes, fps, time_interval)
        dfs[name] = df

        lin, = ax0.plot('Time(s)', 'Mean(IoU)', data=df,
                            linewidth=1.25, alpha=0.75, color=color)
        mark, = ax0.plot('Time(s)', 'Mean(IoU)', data=df,
                             linestyle='', marker='o', markersize=4, color=color)
        legends.append((lin, mark))

    ax0.legend(legends, names, loc=9, ncol=2)
    ax0.xaxis.set_ticklabels([])

    #for c in range(len(classes)):
    #    axes[c].set_ylim(0, 1.1)
    #    axes[c].plot('Time(s)', classes[c] + '(IoU)', data=df_ada, linewidth=1.25)
    #    axes[c].legend()

    lin, = ax1.plot('Time(s)', 'Updates', data=dfs['JITNet 0.8'],
                            linewidth=1.25, alpha=0.75, color=c4)
    mark, = ax1.plot('Time(s)', 'Updates', data=dfs['JITNet 0.8'],
                             linestyle='', marker='o', markersize=4, color=c4)

    #ax1.legend([(lin, mark)], ['Updates'])

    updates_list = dfs['JITNet 0.8']['Updates'].tolist()
    sorted_idx = np.argsort(-np.array(updates_list))

    sorted_idx = sorted_idx[:num_images]

    picked_x = []
    picked_y = []
    for idx in sorted_idx:
        picked_y.append(updates_list[idx])
        picked_x.append(idx * time_interval)
    ax1.plot(picked_x, picked_y, linestyle='', marker='o', markersize=4, color=c2)
    vis_frames = [ int((idx + 0.5) * time_interval * fps) for idx in sorted_idx ]
    frame_dict = get_frames(s, vis_frames)

    ax_image = plt.subplot(gs[0, 2])
    ax_image.imshow(frame_dict[vis_frames[0]])
    ax_image.axis('off')
    ax_image = plt.subplot(gs[0, 3])
    ax_image.imshow(frame_dict[vis_frames[1]])
    ax_image.axis('off')
    ax_image = plt.subplot(gs[0, 4])
    ax_image.imshow(frame_dict[vis_frames[2]])
    ax_image.axis('off')
    ax_image = plt.subplot(gs[1, 2])
    ax_image.imshow(frame_dict[vis_frames[3]])
    ax_image.axis('off')
    ax_image = plt.subplot(gs[1, 3])
    ax_image.imshow(frame_dict[vis_frames[4]])
    ax_image.axis('off')
    ax_image = plt.subplot(gs[1, 4])
    ax_image.imshow(frame_dict[vis_frames[5]])
    ax_image.axis('off')

    plt.savefig(os.path.join('plots', s + '.png'), dpi=300, bbox_inches='tight')
    plt.close()
