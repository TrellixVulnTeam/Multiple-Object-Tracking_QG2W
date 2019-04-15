import sys
sys.path.append('./utils')
import full_segment_iou

v = [
 ('JITNet 2xdepth update every high res', './results/hockey_high_res_large_1_10000.npy'),
 ('JITNet update every high res', './results/hockey_high_res_upper_1_10000.npy'),
 ('JITNet update every', './results/hockey_upper_1_10000.npy'),
 ('JITNet update every low res', './results/hockey_low_res_1_10000.npy'),
 ('JITNet update every half filters', './results/hockey_half_filters_1_10000.npy'),
 ('JITNet adaptive aggresive', './results/hockey_adaptive_25_10000.npy'),
 ('JITNet adaptive', './results/hockey_adam_adaptive_4_25_10000.npy'),
 ('JITNet adaptive half filters', './results/hockey_half_filters_adaptive_4_25_10000.npy'),
 ('JITNet adaptive aggresive multi half filters', './results/hockey_multi_high_adaptive_4_25_10000.npy'),
 ('JITNet adaptive multi half filters', './results/hockey_multi_adaptive_4_25_10000.npy'),
 ('JITNet adaptive no decay', './results/hockey_no_decay_l1.0_adaptive_4_25_10000.npy'),
 ('JITNet pretrain 1000', './results/hockey_cutoff_1_10000.npy'),
 ('JITNet pretrain 500', './results/hockey_cutoff_500_1_10000.npy'),
 ('JITNet fixed', './results/hockey_25_20000.npy'),
 ('JITNet warmup', './results/hockey_warmup_25_20000.npy'),
 ]

full_segment_iou.make_table(v, 10000, './results/schedules.csv', [])
