from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob

import numpy as np
import tensorflow as tf
import time
import cv2
import random

from tensorflow.contrib import slim
from tensorflow.python.ops import control_flow_ops

sys.path.append(os.path.realpath('./src'))
sys.path.append(os.path.realpath('./datasets'))
import res_seg
import res_seg_dual

from mask_rcnn_tfrecords import get_dataset, batch_segmentation_masks,\
                                visualize_masks
from mask_rcnn_stream import MaskRCNNMultiStream, MaskRCNNSequenceStream
import video_distillation

sys.path.append('./models/research/slim/deployment')
import model_deploy as model_deploy

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 300,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer('startup_delay_steps', 15,
                            'Number of training steps between replicas startup.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'Directory containing the dataset.')

tf.app.flags.DEFINE_string(
    'sequence', None, 'Name of the sequence.')

tf.app.flags.DEFINE_integer(
    'height', 720, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'width', 1280, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_frames', 100000,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('cutoff_frame', None,
                            'Frame after which the training is cutoff.')

tf.app.flags.DEFINE_integer('warmup_frames', 0,
                            'Frames for warmup.')

tf.app.flags.DEFINE_string(
    'detections_prefix', None, 'Path to video file.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_boolean(
    'use_seperable_convolution', False,
    'Use a seperable convolution block.')

tf.app.flags.DEFINE_float(
    'filter_depth_multiplier', 1.0,
    'Filter depth multipler for encoder.')

tf.app.flags.DEFINE_integer(
    'num_units', 1,
    'Number of units in each ressep block.')

tf.app.flags.DEFINE_float(
    'scale', 1.0, 'Input scale factor')

tf.app.flags.DEFINE_integer(
    'foreground_weight', 10,
    'Weights for foreground objects.')

tf.app.flags.DEFINE_integer(
    'background_weight', 10,
    'Weights for background objects.')

tf.app.flags.DEFINE_boolean(
    'use_batch_norm', True,
    'Use batch normalization.')

tf.app.flags.DEFINE_boolean(
    'only_train_bnorm', False,
    'Only train the batch norm layers.')

tf.app.flags.DEFINE_integer(
    'training_stride', 25,
    'Stride at which training is done.')

tf.app.flags.DEFINE_integer(
    'max_updates', 4,
    'Max number of iterations on each sample.')

tf.app.flags.DEFINE_integer(
    'replay_buffer_size', 500,
    'Size of the replay buffer.')

tf.app.flags.DEFINE_integer(
    'replay_samples', 1,
    'Number of samples to use from the replay buffer.')

tf.app.flags.DEFINE_boolean(
    'online_train', True,
    'Perform online training.')

tf.app.flags.DEFINE_boolean(
    'adaptive_schedule', False,
    'Use adaptive training schedule.')

tf.app.flags.DEFINE_integer(
    'inference_stride', 10,
    'Stride at which inference is done.')

tf.app.flags.DEFINE_boolean(
    'no_summary', True,
    'Do not compute or write summaries.')

tf.app.flags.DEFINE_string(
    'stats_path', '',
    'If set, will output stats to stats_path')

tf.app.flags.DEFINE_string('video_out_path',
        None, 'Path to output video file.')

tf.app.flags.DEFINE_boolean(
    'scale_boxes', True,
    'Scale the boxes to image height and width.')

tf.app.flags.DEFINE_float(
    'accuracy_threshold', 0.5,
    'Loss threshold at which update stride is halved.')

tf.app.flags.DEFINE_float(
    'confidence_threshold', 0.05,
    'Loss threshold at which update stride is halved.')

tf.app.flags.DEFINE_boolean(
    'train_dual', False,
    'Train a dual architecture.')

tf.app.flags.DEFINE_boolean(
    'focal_loss', False,
    'Use focal loss.')

FLAGS = tf.app.flags.FLAGS

def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            #use_nesterov=True,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer

def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                     for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)

def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def resseg_model(input, height, width, scale, weight_decay,
                 use_seperable_convolution, num_classes,
                 filter_depth_multiplier=1, num_units=1,
                 is_training=True, use_batch_norm=True,
                 dual_model=False):
    model = res_seg
    if dual_model:
        model = res_seg_dual
    with slim.arg_scope(model.ressep_arg_scope(weight_decay=weight_decay,
                                                       use_batch_norm=use_batch_norm)):
        net, end_points = model.ressep_factory(
                                input,
                                use_seperable_convolution = use_seperable_convolution,
                                filter_depth_multiplier = filter_depth_multiplier,
                                is_training = is_training,
                                use_batch_norm = use_batch_norm,
                                num_units = num_units,
                                num_classes = num_classes,
                                scale = scale)

        return net, end_points

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
        global_step = slim.create_global_step()

    video_files = []
    detections_paths = []

    sequence_to_video_list = \
            video_distillation.get_sequence_to_video_list(
                    FLAGS.dataset_dir,
                    FLAGS.dataset_dir,
                    video_distillation.video_sequences)

    print(sequence_to_video_list, FLAGS.sequence)
    assert(FLAGS.sequence in sequence_to_video_list)
    sequence_path = os.path.join(FLAGS.dataset_dir, FLAGS.sequence)

    assert(os.path.isdir(sequence_path))
    assert(FLAGS.sequence in video_distillation.sequence_to_class_groups)

    for s in sequence_to_video_list[FLAGS.sequence]:
        video_files.append(os.path.join(sequence_path, s[0]))
        detections_paths.append(os.path.join(sequence_path, s[1]))
        break

    class_groups = video_distillation.sequence_to_class_groups[FLAGS.sequence]

    print(video_files)
    print(detections_paths)
    print(class_groups)

    class_groups = [ [video_distillation.detectron_classes.index(c) for c in g] \
            for g in class_groups ]
    num_classes = len(class_groups) + 1

    with tf.device('/device:GPU:0'):

        # batch_image = tf.placeholder(tf.uint8, (1,
        #                                         FLAGS.height,
        #                                         FLAGS.width, 3))
        # labels = tf.placeholder(tf.uint8, (1,
        #                                    FLAGS.height,
        #                                    FLAGS.width))
        batch_image = tf.ones((1, FLAGS.height, FLAGS.width, 3), dtype=tf.uint8)
        labels = tf.ones((1, FLAGS.height, FLAGS.width), dtype=tf.uint8)

    def clone_fn(batch_image, labels):
        #low_weights = tf.fill(tf.shape(labels), FLAGS.background_weight)

        #high_weights = tf.fill(tf.shape(labels), FLAGS.foreground_weight)

        #label_weights = tf.where(label_weights_in > 0,
        #                         high_weights, low_weights)

        batch_image = tf.to_float(batch_image)

        labels = tf.cast(labels, tf.int32)
        labels.set_shape([1, FLAGS.height, FLAGS.width])

        batch_image.set_shape([1, FLAGS.height, FLAGS.width, 3])

        logits, end_points = resseg_model(batch_image, FLAGS.height,
                                          FLAGS.width, FLAGS.scale, FLAGS.weight_decay,
                                          FLAGS.use_seperable_convolution,
                                          num_classes,
                                          is_training=True,
                                          dual_model=FLAGS.train_dual,
                                          use_batch_norm=FLAGS.use_batch_norm,
                                          num_units=FLAGS.num_units,
                                          filter_depth_multiplier=FLAGS.filter_depth_multiplier)

        one_hot_labels = tf.one_hot(labels, num_classes)

        if FLAGS.focal_loss:
            probs_true = tf.reduce_max(tf.nn.softmax(logits) * one_hot_labels, axis=3)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,
                                                                       logits=logits,
                                                                       dim=-1,
                                                                       name='xentropy')
            gamma = 2
            clipped_weights = tf.pow(tf.clip_by_value((1.-probs_true), 0.1, 0.9), gamma)
            focal_loss = 10 * tf.reduce_mean(cross_entropy + (clipped_weights * cross_entropy))
            tf.losses.add_loss(focal_loss)
        else:
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels,
                                                                   logits,
                                                                   weights=label_weights,
                                                                   scope='xentropy')
            tf.losses.add_loss(cross_entropy)

        #probs = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, axis=3)
        predictions = tf.cast(predictions, tf.uint8)

        probs = tf.one_hot(predictions, num_classes)
        fp = tf.reduce_sum(probs * (1. - one_hot_labels), axis=[0, 1, 2])
        tp = tf.reduce_sum(probs * one_hot_labels, axis=[0, 1, 2])
        fn = tf.reduce_sum((1. - probs) * one_hot_labels, axis=[0, 1, 2])

        eps = 100
        numerator = tp + eps
        denominator = tp + fp + fn + eps
        class_scores = numerator / denominator

        #class_scores = tf.zeros([num_classes], tf.float32)

        return end_points, predictions, class_scores

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, \
                                        [batch_image, labels])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, global_step)
    else:
        moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
        learning_rate = FLAGS.learning_rate
        optimizer = _configure_optimizer(learning_rate)

    if FLAGS.sync_replicas:
        # If sync_replicas is enabled, the averaging will be done in the chief
        # queue runner.
        optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables,
          replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
          total_num_replicas=FLAGS.worker_replicas)
    elif FLAGS.moving_average_decay:
        # Update ops executed locally by trainer.
        update_ops.append(variable_averages.apply(moving_average_variables))

    end_points, predictions, class_scores = clones[0].outputs

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    if FLAGS.only_train_bnorm:
        variables_to_train = tf.contrib.framework.filter_variables(variables_to_train,
                                                                   include_patterns=['BatchNorm', 'logits'])

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(clones,
                                                    optimizer,
                                                    var_list=variables_to_train)
    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                      name='train_op')

    if FLAGS.sync_replicas:
        sync_optimizer = opt
        startup_delay_steps = 0
    else:
        sync_optimizer = None
        startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps

    ###########################
    # Kicks off the training. #
    ###########################
    #with tf.contrib.tfprof.ProfileContext(FLAGS.train_dir) as pctx:

    print(video_files, detections_paths)
    input_streams = MaskRCNNSequenceStream(video_files, detections_paths,
                                           start_frame=0, stride=1)

    init_fn = _get_init_fn()
    replay_buffer = {}

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if init_fn is not None:
            init_fn(sess)

        curr_frame = 0
        training_strides = [4, 8, 16, 32]
        curr_stride_idx = 0

        zero_input_labels = np.zeros((1, FLAGS.height, FLAGS.width),
                                      np.uint8)
        zero_input_weights = np.zeros((1, FLAGS.height, FLAGS.width),
                                      np.bool)
        inference_str = ""
        training_str = ""
        stats_str = ""
        stride_str = ""

        num_teacher_samples = 0
        num_updates = 0
        replay_buffer_idx = 0

        for frame, boxes, classes, scores, masks, num_objects, frame_id in input_streams:
            if curr_frame > FLAGS.max_frames:
                break

            frame = cv2.resize(frame, (FLAGS.width, FLAGS.height))

            frame = np.expand_dims(frame, axis=0)
            boxes = np.expand_dims(boxes, axis=0)
            classes = np.expand_dims(classes, axis=0)
            scores = np.expand_dims(scores, axis=0)
            masks = np.expand_dims(masks, axis=0)
            num_objects = np.expand_dims(num_objects, axis=0)

            preds = None
            summary_str = ""
            online_train = FLAGS.online_train

            if FLAGS.cutoff_frame is not None and curr_frame > FLAGS.cutoff_frame:
                online_train = False

            if FLAGS.adaptive_schedule:
                train_stride = training_strides[curr_stride_idx]
            else:
                train_stride = FLAGS.training_stride

            if (curr_frame % train_stride == 0 or
                curr_frame < FLAGS.warmup_frames) and online_train:
                num_teacher_samples = num_teacher_samples + 1
                labels_vals, label_weights_vals = batch_segmentation_masks(1,
                                                    (FLAGS.height, FLAGS.width),
                                                    boxes, classes, masks, scores,
                                                    num_objects, True,
                                                    class_groups,
                                                    scale_boxes=FLAGS.scale_boxes)

                replay_buffer[replay_buffer_idx] = frame, labels_vals, label_weights_vals
                replay_buffer_idx = (replay_buffer_idx + 1)%FLAGS.replay_buffer_size

                buffer_idxs = replay_buffer.keys()

                frame_list = [frame]
                labels_list = [labels_vals]
                label_weights_list = [label_weights_vals]

                for r in range(FLAGS.replay_samples):
                    idx = random.choice(buffer_idxs)
                    rframe, rlabels_vals, rlabel_weights_vals = replay_buffer[idx]
                    frame_list.append(rframe)
                    labels_list.append(rlabels_vals)
                    label_weights_list.append(rlabel_weights_vals)

                in_images = np.concatenate(frame_list, axis=0)
                labels_vals = np.concatenate(labels_list, axis=0)
                label_weights_vals = np.concatenate(label_weights_list, axis=0)

                start = time.time()
                curr_updates = 0
                while True:
                    start_in = time.time()

                    step, _, loss = sess.run([global_step, train_tensor,
                                              total_loss])
                    end_in = time.time()
                    print('one update time:', end_in - start_in)

                    prof = tf.profiler.profile(tf.get_default_graph(),
                                options=tf.profiler.ProfileOptionBuilder.float_operation())
                    print('Total flops:', prof.total_float_ops)

                    num_updates = num_updates + 1
                    curr_updates = curr_updates + 1
                    cls_scores = np.zeros((1))

                end = time.time()

                if min(cls_scores) > FLAGS.accuracy_threshold:
                    curr_stride_idx = min(curr_stride_idx + 1, len(training_strides) - 1)
                else:
                    curr_stride_idx = max(curr_stride_idx - 1, 0)

                training_str = "training: {0:.5f}s minIoU: {1:.5f}".format(end - start, min(cls_scores))

                stride_str = ""
                if FLAGS.adaptive_schedule:
                    stride_str = "num_teacher_samples: %d num_updates: %d stride: %d"%(num_teacher_samples,
                                                                                       num_updates,
                                                                                       training_strides[curr_stride_idx])

            elif curr_frame % FLAGS.inference_stride == 0:
                start = time.time()
                step, preds = sess.run([global_step, predictions])
                end = time.time()
                inference_str = "inference: {0:.5f}s".format(end - start)

            frame_str = "frame: {0:05d}".format(curr_frame)
            print(" ".join([frame_str, training_str, inference_str, stats_str,
                            stride_str, summary_str]), end="\r", file=sys.stderr)
            curr_frame = curr_frame + 1
        print(file=sys.stderr)

if __name__ == '__main__':
  tf.app.run()
