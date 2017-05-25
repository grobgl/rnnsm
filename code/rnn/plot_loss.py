import numpy as np
from tensorflow.python.summary.event_accumulator import EventAccumulator

import sys
sys.path.insert(0, '../utils')
from plot_format import *
from seaborn import apionly as sns

path_no_dropout = '../../logs/rnn_new/08_larger_lr0.01_inp11_nsess64_hiddenNr16/events.out.tfevents.1495619607.georg-xps-15'
path_out_dropout = '../../logs/rnn_new/13_08_w_dropout_lr0.01_inp11_nsess64_hiddenNr16/events.out.tfevents.1495633278.georg-xps-15'
path_all_dropout = '../../logs/rnn_new/14_08_w_rec_dropout_lr0.01_inp11_nsess64_hiddenNr16/events.out.tfevents.1495634315.georg-xps-15'

def plot_tensorflow_log(path, width=1, height=None):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        # 'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())
    # return event_acc

    training_loss =   event_acc.Scalars('loss')
    validation_loss = event_acc.Scalars('val_loss')

    steps = len(training_loss)
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_loss[i][2] # value
        y[i, 1] = validation_loss[i][2]

    fig, ax = newfig(width, height)

    ax.plot(x, y[:,0], label='training loss')
    ax.plot(x, y[:,1], label='validation loss')

    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    fig.show()

