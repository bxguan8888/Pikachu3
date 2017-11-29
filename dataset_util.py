#import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.image as image
import numpy as np

data_shape = 256
batch_size = 32


def get_iterators(data_shape, batch_size):
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_train.rec',
        path_imgidx='./data/pikachu_train.idx',
        shuffle=False,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class


def get_iterators_2(data_shape, batch_size):
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = mx.io.ImageRecordIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_train.rec',
        path_imgidx='./data/pikachu_train.idx',
        label_width=9,
        shuffle=False,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = mx.io.ImageRecordIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='./data/pikachu_val.rec',
        label_width=9,
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class


if __name__ == "__main__":
    train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
    train_data2, test_data2, class_names2, num_class2 = get_iterators_2(data_shape, batch_size)
    i = 0

    for batch in train_data:
        i += 1
        print '%d batch: %s' % (i, str(batch))


