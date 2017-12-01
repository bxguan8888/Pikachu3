from __future__ import absolute_import

import mxnet as mx
import numpy as np

# coding: utf-8
"""TensorBoard functions that can be used to log various status during epoch."""
import logging


class LogMetricsCallback(object):
    def __init__(self, logging_dir, prefix=None):
        self.prefix = prefix
        self.itr = 0
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, name_value):
        """Callback to log training speed and metrics in TensorBoard."""
        if name_value is None:
            return
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(name, value, self.itr)
        self.itr += 1
