import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys, os
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

results_dir = '/n/pana/scratch/ravi/video_distillation_final/'
#varaint_suffix = '_stride_8_frame_30000_thresh_0.8_lr_0.01'
varaint_suffix = '_stride_8_frame_30000_thresh_0.9_lr_0.01'
for s in video_distillation.video_sequences_stable:
    class_names = video_distillation.sequence_to_class_groups_stable[s]
    classes = ['background'] + [ '_'.join(g) for g in class_names ]
    results_path = os.path.join(results_dir, s + varaint_suffix + '.npy')
    df = make_data_frame(results_path, classes, 25, 5)
    num_plots = len(classes) + 1
    fig, axes = plt.subplots(num_plots, 1, sharex=True)

    for c in range(len(classes)):
        axes[c].set_ylim(0, 1.1)
        axes[c].plot('Time(s)', classes[c] + '(IoU)', data=df, linewidth=1.25)
        axes[c].legend()

    axes[-1].plot('Time(s)', 'Updates', data=df)
    axes[-1].legend()

    plt.savefig(os.path.join('plots', s + varaint_suffix + '.png'))
    plt.close()
