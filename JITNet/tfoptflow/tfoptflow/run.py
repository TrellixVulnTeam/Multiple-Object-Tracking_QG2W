from __future__ import absolute_import, division, print_function
import sys
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import plot_img_pairs_w_flows
from optflow import flow_to_img, flow_write_as_png
import argparse
import cv2
import imageio
import numpy as np
from copy import deepcopy

# import stream utility from JITNet utils
sys.path.append('../../utils')
from stream import VideoInputStream


CKPT_PATH = '../../../pwcnet/pwcnet_lg/pwcnet.ckpt-595000'

def main():
    # parse input and output file paths
    parser = argparse.ArgumentParser(description='Run PWCNet on video.')
    parser.add_argument('input', help='input video file')
    parser.add_argument('output', help='output video file')
    args = parser.parse_args()

    video = VideoInputStream(args.input, start_frame=0)
    out_video = cv2.VideoWriter(args.output,
                                cv2.VideoWriter_fourcc(*'H264'),
                                video.rate, (2 * video.width, video.height))

    # set NN test options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = CKPT_PATH
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = ['/device:GPU:0']
    nn_opts['controller'] = '/device:GPU:0'
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # since the model generates flow padded to multiples of 64, reduce back to 1080p
    # TODO: this should be whatever the input video width and height is
    nn_opts['adapt_info'] = (1, video.height, video.width, 2)

    nn = ModelPWCNet(mode='test', options=nn_opts)

    prev = None
    for i, im in enumerate(video):
        print(i)
        if i == 0:
            prev = im
            continue
        # run flow on prev and im
        img_pair = [(prev, im)]
        pred_labels = nn.predict_from_img_pairs(img_pair, batch_size=1, verbose=False)
        pred = pred_labels[0]

        # show results
        flow_im = flow_to_img(pred, flow_mag_max=100)

        vis_image = np.concatenate((im, flow_im), axis=1)
        out_video.write(vis_image)

        prev = im

if __name__ == '__main__':
    main()
