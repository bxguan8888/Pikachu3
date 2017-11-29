import logging
import time
from logging import debug as DEBUG

import mxnet as mx
from easydict import EasyDict as edict
from mxnet.contrib.symbol import MultiBoxPrior
from mxnet.contrib.symbol import MultiBoxTarget

from dataset_util import get_iterators
from resnet import get_resnet_symbol

from mxnet.model import BatchEndParam
from mxnet.base import _as_list

from metric import LogMetricsCallback
import time
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def class_predictor(sym, num_anchors, num_classes):
    """return a layer to predict classes"""
    if num_classes == 1:
        # set num_classes to 0 if only detect foreground and background
        num_classes = 1
    return mx.sym.Convolution(data=sym, num_filter=num_anchors * (num_classes + 1),
                              kernel=(3, 3), pad=(1, 1), stride=(1, 1), no_bias=True)


def box_predictor(sym, num_anchors):
    """return a layer to predict delta locations"""
    return mx.sym.Convolution(data=sym, num_filter=num_anchors * 4, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                              no_bias=True)


def verify_shape(sym, data_shape, label_shape=None):
    if label_shape == None:
        args_shape, out_shape, aux_shape = sym.infer_shape(data=data_shape)
    else:
        args_shape, out_shape, aux_shape = sym.infer_shape(data=data_shape, label=label_shape)
    return "The output shape is {}".format(out_shape)


def save_plot(sym, name, data_shape, label_shape=None):
    if label_shape is None:
        plot = mx.viz.plot_network(sym, shape={'data': data_shape})
    else:
        plot = mx.viz.plot_network(sym, shape={'data': data_shape, 'label': label_shape})
    plot.render(name)


def get_all_bn_layers(final_sym, layer_list):
    layers = []
    all_layers = final_sym.get_internals()
    for name in layer_list:
        layers.append(all_layers[name])
    return layers


def training_targets(default_anchors, class_predicts, labels):
    class_predicts = mx.sym.transpose(class_predicts, axes=(0, 2, 1))
    z = MultiBoxTarget(anchor=default_anchors, label=labels, cls_pred=class_predicts)
    box_target = z[0]
    box_mask = z[1]
    cls_target = z[2]
    return box_target, box_mask, cls_target


def flatten_prediction(pred):
    return mx.sym.flatten(mx.sym.transpose(pred, axes=(0, 2, 3, 1)))


def concat_predictions(preds):
    return mx.sym.concat(*preds, dim=1)


def callback(metric_list, callback_list):
    for metric in metric_list:
        batch_end_params = BatchEndParam(epoch=epoch,
                                         nbatch=i,
                                         eval_metric=metric,
                                         locals=locals())
        for callback in _as_list(batch_end_callback):
            callback(batch_end_params)


class FocalLoss():
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        self._batch_axis = batch_axis
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, output, label):
        output = mx.sym.softmax(output)
        pt = mx.sym.pick(output, label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pt) ** self._gamma) * mx.sym.log(pt)
        return mx.sym.mean(loss, axis=self._batch_axis, exclude=True)


class SmoothL1Loss():
    def __init__(self, batch_axis=0, **kwargs):
        self._batch_axis = batch_axis

    def hybrid_forward(self, output, label, mask):
        loss = mx.sym.smooth_l1((output - label) * mask, scalar=1.0)
        return mx.sym.mean(loss, self._batch_axis, exclude=True)


def SSD_builder(args):
    # get resnet symbol
    resnet = get_resnet_symbol(num_classes=1, num_layers=18, image_shape=args.image_shape)
    last_relu = resnet.get_internals()['relu1_output']
    data = resnet.get_internals()['data']
    label = mx.sym.Variable('label')
    # extract the layers right before downsampling
    multiscalelayers_name = args.multiscalelayers_name
    multiscalelayers_layers = get_all_bn_layers(last_relu, multiscalelayers_name)

    # Build the predict boxes
    predicted_boxes = []
    predicted_classes = []
    default_anchors = []
    # Add relu+conv
    for layer, size, ratio in zip(multiscalelayers_layers, args.anchor_sizes, args.anchor_ratios):
        DEBUG("Verify the output shape of the layer: {}".format(layer.name))
        DEBUG("Before->" + verify_shape(layer, (1, 3) + tuple(args.data_shape)))
        relu = mx.sym.Activation(data=layer, act_type='relu', name=layer.name + '_relu')
        boxes = box_predictor(relu, args.num_anchors)
        classes = class_predictor(relu, args.num_anchors, args.num_classes)
        anchors = MultiBoxPrior(layer, sizes=size, ratios=ratio, clip=True)
        DEBUG("After Box->" + verify_shape(boxes, (1, 3) + tuple(args.data_shape)))
        DEBUG("After Class->" + verify_shape(classes, (1, 3) + tuple(args.data_shape)))
        DEBUG("After Anchor->" + verify_shape(anchors, (1, 3) + tuple(args.data_shape)))
        predicted_boxes.append(flatten_prediction(boxes))
        predicted_classes.append(flatten_prediction(classes))
        default_anchors.append(anchors)
    all_anchors = concat_predictions(default_anchors)
    all_classes_pred = mx.sym.reshape(concat_predictions(predicted_classes), shape=(0, -1, args.num_classes + 1))
    all_boxes_pred = concat_predictions(predicted_boxes)
    DEBUG("All anchors->" + verify_shape(all_anchors, (1, 3) + tuple(args.data_shape)))
    DEBUG("All classes->" + verify_shape(all_classes_pred, (1, 3) + tuple(args.data_shape)))
    DEBUG("All boxes->" + verify_shape(all_boxes_pred, (1, 3) + tuple(args.data_shape)))

    # setup groundtruth label
    box_target, box_mask, cls_target = training_targets(all_anchors, all_classes_pred, label)
    DEBUG("box_target->" + verify_shape(box_target, (1, 3) + tuple(args.data_shape), (1, 1, 5)))
    DEBUG("box_mask->" + verify_shape(box_mask, (1, 3) + tuple(args.data_shape), (1, 1, 5)))
    DEBUG("cls_target->" + verify_shape(cls_target, (1, 3) + tuple(args.data_shape), (1, 1, 5)))

    cls_loss = FocalLoss()
    box_loss = SmoothL1Loss()

    loss1 = cls_loss.hybrid_forward(all_classes_pred, cls_target)
    loss2 = box_loss.hybrid_forward(all_boxes_pred, box_target, box_mask)
    loss = loss1 + loss2

    DEBUG("Final loss->" + verify_shape(loss, (1, 3) + tuple(args.data_shape), (1, 1, 5)))

    loss_make = mx.sym.MakeLoss(loss)
    output = mx.sym.Group([loss_make,
                           mx.sym.BlockGrad(all_classes_pred), mx.sym.BlockGrad(cls_target),
                           mx.sym.BlockGrad(all_boxes_pred), mx.sym.BlockGrad(box_target),
                           mx.sym.BlockGrad(box_mask)])
    return output


if __name__ == "__main__":
    # setup configure setting
    args = edict()
    args.do_check_point = False
    args.log_to_tensorboard = True
    args.ctx = mx.gpu(0)

    args.batch_size = 32
    args.data_shape = (256, 256)
    args.num_classes = 1
    args.image_shape = "3, 256,256"
    args.save_plot = False
    args.infer_shape = True
    args.num_anchors = 4
    args.multiscalelayers_name = ["bn0_output", "stage2_unit1_bn1_output",
                                  "stage3_unit1_bn1_output", "stage4_unit1_bn1_output", "bn1_output"]
    args.anchor_sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
    args.anchor_ratios = ([[1, 2, .5]] * 5)
    args.epoch = 100
    args.learning_rate = 0.001
    args.optimizer = 'adam'
    args.log_interval = 2

    args.save_name = "pikachu"

    args.optimizer = "adam"
    args.prefix = "SSD_resnet"

    # get training/testing dataset
    train_data, test_data, class_names, num_class = \
        get_iterators(args.data_shape[0], args.batch_size)
    
    train_data.reshape(label_shape=(3, 5))
    test_data.reshape(label_shape=(3, 5))

    train_data.provide_data = [('data', (32, 3, 256, 256))]
    loss = SSD_builder(args)

    # setup checkpoint callback function
    checkpoint = mx.callback.do_checkpoint(args.save_name)

    # set optimizer
    optimizer_params = {
        'learning_rate': args.learning_rate,
    }

    tme = time.time()
    batch_end_callback = [
        LogMetricsCallback('logs/train-' + str(tme)),
    ]
    val_batch_end_callback = [
        LogMetricsCallback('logs/val-' + str(tme)),
    ]

    mod = mx.mod.Module(context=args.ctx,
                        symbol=loss,
                        data_names=['data'],
                        label_names=['label'])
    mod.bind(data_shapes=[('data', (args.batch_size, 3, args.data_shape[0], args.data_shape[1]))],
             label_shapes=train_data.provide_label,
             for_training=True)
    mod.init_params(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2.))
    mod.init_optimizer(optimizer=args.optimizer, optimizer_params=(('learning_rate', args.learning_rate),))

    # setup metric
    cls_metric = mx.metric.Accuracy()
    box_metric = mx.metric.MAE()  # measure absolute difference between prediction and target

    for epoch in range(args.epoch):
        train_data.reset()
        cls_metric.reset()
        box_metric.reset()
        tic = time.time()

        for i, batch in enumerate(train_data):
            btic = time.time()
            mod.forward(batch, is_train=True)
            preds = mod.get_outputs(merge_multi_context=True)
            loss, all_classes_pred, cls_target, all_boxes_pred, box_target, box_mask = preds
            cls_metric.update([cls_target], [mx.nd.transpose(all_classes_pred, (0, 2, 1))])
            box_metric.update([box_target], [all_boxes_pred * box_mask])
            if args.log_to_tensorboard:
                callback([cls_metric, box_metric], batch_end_callback)
            mod.backward()
            mod.update()

            if (i + 1) % args.log_interval == 0:
                name1, val1 = cls_metric.get()
                name2, val2 = box_metric.get()
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f'
                      % (epoch, i, args.batch_size / (time.time() - btic), name1, val1, name2, val2))
             # end of epoch logging

        name1, val1 = cls_metric.get()
        name2, val2 = box_metric.get()
        print('[Epoch %d] training: %s=%f, %s=%f' % (epoch, name1, val1, name2, val2))
        print('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))
        if args.do_check_point:
            mod.save_checkpoint(args.prefix, epoch)
        
        # validation
        test_data.reset()
        cls_metric.reset()
        box_metric.reset()

        i = 0
        for i, batch in enumerate(test_data):
            mod.forward(batch, is_train=False)
            preds = mod.get_outputs(merge_multi_context=True)
            loss, all_classes_pred, cls_target, all_boxes_pred, box_target, box_mask = preds
            cls_metric.update([cls_target], [mx.nd.transpose(all_classes_pred, (0, 2, 1))])
            box_metric.update([box_target], [all_boxes_pred * box_mask])
            if args.log_to_tensorboard:
                callback([cls_metric, box_metric], val_batch_end_callback)

        name1, val1 = cls_metric.get()
        name2, val2 = box_metric.get()
        print('[Epoch %d] testing: %s=%f, %s=%f' % (epoch, name1, val1, name2, val2))
