# script to plot the sequence data
# Author: LIU Jianbang @ RPAI Lab, EE, CUHK
# Date: 2021.06.19

import numpy as np
import matplotlib.pyplot as plt

COLOR = {
    0: "g",     # green for ground truth
    1: "b",     # blue for prediction
}

LINE = {
    0: "-",     # solid line for ground truth
    1: "--",    # dash line for prediction
}


def show_pred_and_gt(ax, y_gt, y_pred):
    """"""
    # plot gt
    ax.plot(y_gt[:, 0], y_gt[:, 1], COLOR[0]+"x"+LINE[0])

    # plot preds
    for pred in y_pred:
        ax.plot(pred[:, 0], pred[:, 1], COLOR[1]+LINE[1])

