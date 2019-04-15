import sys, os
sys.path.append('./utils')
sys.path.append('./datasets')
import full_segment_iou
import video_distillation

ablation_list = ['biking1', 'samui_murphys1', 'driving1', 'dogs2', \
                 'figure_skating1', 'kabaddi1' ]

for s in ablation_list:
    results_dirs = ['/n/pana/scratch/ravi/video_distillation_final',
                    '/n/pana/scratch/ravi/video_distillation_ablation',
                    '/n/pana/scratch/ravi/video_distillation_ablation',
                    '/n/pana/scratch/ravi/video_distillation_ablation',
                    '/n/pana/scratch/ravi/video_distillation_ablation',
                    '/n/pana/scratch/ravi/video_distillation_ablation',
                    '/n/pana/scratch/ravi/video_distillation_ablation']

    variants = ['_stride_8_frame_30000_thresh_0.8_lr_0.01',
                '_stride_8_frame_30000_th_0.8_lr_0.01_mu_4_ms_8',
                '_stride_8_frame_30000_th_0.8_lr_0.01_mu_16_ms_8',
                '_stride_8_frame_30000_th_0.8_lr_0.001_mu_8_ms_8',
                '_stride_8_frame_30000_th_0.8_lr_0.1_mu_8_ms_8',
                '_stride_8_frame_30000_th_0.8_lr_0.01_mu_8_ms_4',
                '_stride_8_frame_30000_th_0.8_lr_0.01_mu_8_ms_16',
                ]
    variant_names = ['JITNet 0.8',
                     'JITNet 0.8 max updates 4',
                     'JITNet 0.8 max updates 16',
                     'JITNet 0.8 lr 0.001',
                     'JITNet 0.8 lr 0.1',
                     'JITNet 0.8 min stride 4',
                     'JITNet 0.8 min stride 16'
                     ]
    variant_cost_model = ['jitnet', 'jitnet', 'jitnet', 'jitnet', 'jitnet', 'jitnet', 'jitnet']
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
                                os.path.join('./tables_ablation', s + '.csv'), [],
                                start_frame=0)
