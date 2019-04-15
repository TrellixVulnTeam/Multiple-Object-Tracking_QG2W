import sys
sys.path.append('./utils')
import full_segment_iou

v = [
 ('JITNet V1 pretrained', '/n/pana/scratch/ravi/many_cam_train/jackson_hole/stats_1/online_bg_jackson_hole_6_25_120000.npy'),
 ('JITNet V1 scratch', '/n/pana/scratch/ravi/many_cam_train/jackson_hole/stats_1/online_bg_scratch_jackson_hole_6_25_400000.npy'),
 ('JITNet V2 scratch', '/n/pana/scratch/ravi/JITNet/results/online_bg_scratch_jackson_hole_6_25_120000.npy'),
 ('JITNet V2 big scratch', '/n/pana/scratch/ravi/JITNet/results/online_bg_scratch_larger_jackson_hole_6_25_120000.npy'),
 ('JITNet V2 big lr+ scratch', '/n/pana/scratch/ravi/JITNet/results/online_bg_scratch_r0.005_jackson_hole_6_25_120000.npy'),
 ]

full_segment_iou.make_table(v, 1000000, './results/jackson_hole_v1_vs_v2.csv', [])
