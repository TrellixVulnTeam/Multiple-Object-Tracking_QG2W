import sys, os
sys.path.append('./utils')
sys.path.append('./datasets')
import full_segment_iou
import video_distillation

for s in video_distillation.video_sequences_stable:
    results_dirs = [#'/n/pana/scratch/ravi/video_distillation_long_stable',
                    #'/n/pana/scratch/ravi/video_distillation_long_stable',
                    #'/n/pana/scratch/ravi/video_distillation_long_stable',
                    '/n/pana/scratch/ravi/video_distillation_final',
                    '/n/pana/scratch/ravi/video_distillation_final',
                    '/n/pana/scratch/ravi/video_distillation_final',
                    #'/n/pana/scratch/ravi/video_distillation_replay',
                    #'/n/pana/scratch/ravi/video_distillation_replay_correct',
                    #'/n/pana/scratch/ravi/video_distillation_thresh',
                    #'/n/pana/scratch/ravi/video_distillation_no_focal',
                    #'/n/pana/scratch/ravi/video_distillation_large',
                    '/n/pana/scratch/ravi/video_distillation_overfit',
                    '/n/pana/scratch/ravi/video_distillation_half_overfit',
                    '/n/pana/scratch/ravi/flow-30',
                    '/n/pana/scratch/ravi/flow-30small',
                    '/n/pana/scratch/ravi/osvos',
                    '/n/pana/scratch/ravi/osvos_static',
                    #'/n/pana/scratch/ravi/video_distillation_noise',
                    #'/n/pana/scratch/ravi/video_distillation_upper_bound',
                    #'/n/pana/scratch/ravi/video_distillation_previous',
                    #'/n/pana/scratch/ravi/video_distillation_previous'
                    ]
    variants = ['_stride_8_frame_30000_thresh_0.9_lr_0.01',
                '_stride_8_frame_30000_thresh_0.8_lr_0.01',
                '_stride_8_frame_30000_thresh_0.7_lr_0.01',
                #'_stride_25_frame_15000_thresh_0.90_lr_0.01',
                #'_stride_16_frame_15000_thresh_0.75_lr_0.01',
                #'_stride_8_frame_15000_thresh_0.75_lr_0.01',
                #'_stride_8_frame_30000_thresh_0.9_lr_0.01_upper_bound',
                #'_stride_8_frame_30000_thresh_0.8_lr_0.01',
                #'_stride_8_frame_30000_thresh_0.8_lr_0.01',
                #'_stride_8_frame_30000_thresh_0.8_lr_0.01',
                #'_stride_8_frame_30000_thresh_0.7_lr_0.01_upper_bound',
                #'_stride_8_frame_30000_thresh_0.8_lr_0.01_upper_bound',
                '_stride_8_frame_30000_overfit',
                '_stride_8_frame_30000_overfit',
                '',
                '',
                '',
                '',
                #'_stride_8_frame_30000_thresh_0.8_lr_0.01_upper_bound',
                #'_stride_16_frame_15000_pp',
                #'_stride_32_frame_15000_pp'
                ]
    variant_names = ['JITNet 0.9',
                     'JITNet 0.8',
                     'JITNet 0.7',
                     'JITNet Offline Overfit',
                     'JITNet Pretrain',
                     'Flow Slow',
                     'Flow Fast',
                     'OSVOS',
                     'OSVOS static']
    variant_cost_model = ['jitnet', 'jitnet', 'jitnet', 'jitnet', 'jitnet', 'flow', 'flow', 'osvos', 'osvos']
    variant_list = []

    for rdir, var, var_name, var_cost in zip(results_dirs, variants, variant_names,
                                             variant_cost_model):
        results_path = os.path.join(rdir, s + var + '.npy')
        if not os.path.exists(results_path):
            continue

        variant_list.append((var, results_path, var_name, var_cost))

    class_names = video_distillation.sequence_to_class_groups_stable[s]
    class_names = ['background'] + [ '_'.join(g) for g in class_names ]
    #full_segment_iou.make_table(class_names, variant_list, 5400 + 3600,
    #                            os.path.join('./tables_osvos', s + '.csv'), [],
    #                            start_frame=5400)
    full_segment_iou.make_table(class_names, variant_list, 30000,
                                os.path.join('./tables', s + '.csv'), [],
                                start_frame=0)
