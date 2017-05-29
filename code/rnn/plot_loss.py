import numpy as np
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorflow.tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import sys
sys.path.insert(0, '../utils')
from plot_format import *
from seaborn import apionly as sns
import matplotlib.gridspec as gridspec

path_no_reg = '../../logs/rnn_new/39_rnn_explore_lr0.01_inp15_nsess50_hiddenNr10/events.out.tfevents.1495986915.georg-XPS'
path_act_dropout = '../../logs/rnn_new/42_rnn_39_dropoutact0.2_lr0.01_inp15_nsess50_hiddenNr10/events.out.tfevents.1495994380.georg-XPS'
path_unit_norm = '../../logs/rnn_new/44_rnn_39_unit_norm_kernel_lr0.01_inp15_nsess50_hiddenNr10/events.out.tfevents.1496000104.georg-XPS'
path_l2_penalty = '../../logs/rnn_new/45_rnn_39_l2_reg.25_lr0.01_inp15_nsess50_hiddenNr10/events.out.tfevents.1496031031.georg-XPS'
# path_out_dropout = '../../logs/rnn_new/13_08_w_dropout_lr0.01_inp11_nsess64_hiddenNr16/events.out.tfevents.1495633278.georg-xps-15'
# path_all_dropout = '../../logs/rnn_new/14_08_w_rec_dropout_lr0.01_inp11_nsess64_hiddenNr16/events.out.tfevents.1495634315.georg-xps-15'

def plot_tensorflow_log_all(width=1, height=None):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        # 'scalars': 100,
        'histograms': 1
    }

    paths = [path_no_reg, path_act_dropout, path_unit_norm, path_l2_penalty]

    event_accs = [EventAccumulator(p, tf_size_guidance) for p in paths]
    # event_acc.Reload()
    [e.Reload() for e in event_accs]

    # Show all tags in the log file
    #print(event_acc.Tags())
    # return event_acc

    training_loss = [e.Scalars('loss') for e in event_accs]
    validation_loss = [e.Scalars('val_loss') for e in event_accs]
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.025, hspace=.05)

    axs = [0]*4
    # fig, axs[0] = newfig(width, height, gs[0])
    # axs[1] = fig.add_subplot(gs[1])
    # axs[2] = fig.add_subplot(gs[2])
    fig, axs[0] = newfig(width, height, 141)
    axs[1] = fig.add_subplot(142)
    axs[2] = fig.add_subplot(143)
    axs[3] = fig.add_subplot(144)

    for i,ax in enumerate(axs):
        steps = len(training_loss[i])
        x = np.arange(steps)
        y = np.zeros([steps, 2])

        for j in range(steps):
            y[j, 0] = training_loss[i][j][2] # value
            y[j, 1] = validation_loss[i][j][2]

        tr_loss_plt = ax.plot(x, y[:,0], label='training loss')[0]
        val_loss_plt = ax.plot(x, y[:,1], label='validation loss')[0]
        ax.set_ylim((600, 5700))

    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[3].set_yticks([])
    axs[0].set_ylabel('Loss (MSE)')
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    axs[2].set_xlabel('Epoch')
    axs[3].set_xlabel('Epoch')
    axs[0].set_title('No regularisation')
    axs[1].set_title('Dropout')
    axs[2].set_title('Unit norm')
    axs[3].set_title('L2 penalty')

    fig.legend(handles=[tr_loss_plt, val_loss_plt], labels=['training loss','validation loss'], loc='upper center', ncol=3, framealpha=1, bbox_to_anchor=(0.54, 0.82), columnspacing=10)
    # fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.show()

    return


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

