import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
from collections import namedtuple


def parse_args():
    """
    retrieving arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Run GAAL")
    parser.add_argument('--dset', default='mnist57', help='dataset')
    parser.add_argument('--limit', type=int, default=250, help='Total label budget')
    return parser.parse_args()


# def load_data(path):
#     """
#     loading data from the given path.
#     """
#     data = pd.read_table('{path}'.format(path=path), sep=',', header=None)
#     data = data.sample(frac=1).reset_index(drop=True)
#     id = data.pop(0)
#     y = data.pop(1)
#     data_x = data.values
#     data_id = id.values
#     data_y = y.values
#     data_y[data_y == 'nor'] = 1
#     data_y[data_y == 'out'] = 0
#     data_y = data_y.astype(np.int32)
#     return data_x, data_y, data_id


def plot(x,y, dset, limit, save_dir='plots/'):
    """
    plot training history.
    """
    plt.plot(x, y, 'o-')
    plt.grid()
    run_id = generate_run_id()
    plt.savefig(f"./plot/{dset}_{limit}_{run_id}.png")
    plt.show()


def generate_run_id():
    """
    construct the run ID from time
    """
    year = str(datetime.now().year)
    month = '{:02d}'.format(datetime.now().month)
    day = '{:02d}'.format(datetime.now().day)
    hour = '{:02d}'.format(datetime.now().hour)
    minute = '{:02d}'.format(datetime.now().minute)
    second = '{:02d}'.format(datetime.now().second)
    current_run_id = year + month + day + hour + minute + second

    return current_run_id
