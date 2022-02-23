import argparse
import os
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


def plot(x, y, dset, limit, id, rt, save_dir='./plot'):
    """
    plot training history.
    """
    result_dir = f"{save_dir}/{dset}_{limit}_{id}"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    plt.plot(x, y, 'o-')
    plt.grid()
    plt.savefig(f"{result_dir}/{rt}.png")
    # plt.show()
    plt.close()


def plot_err(x, y, err, dset, limit, id, save_dir='./plot'):
    """
    plot training history with errorbar
    """
    result_dir = f"{save_dir}/{dset}_{limit}_{id}"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    plt.errorbar(x, y, yerr=err, capsize=2)
    plt.grid()
    plt.savefig(f"{result_dir}/final_with_err.png")
    plt.show()
    plt.close()


def plot_all(x, gaal_list, random_list, full_list, dset, limit, id, save_dir='./plot'):
    result_dir = f"{save_dir}/{dset}_{limit}_{id}"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    gaal_aver, gaal_sd = np.mean(gaal_list, axis=0), np.std(gaal_list, axis=0)
    random_aver, random_sd = np.mean(random_list, axis=0), np.std(random_list, axis=0)
    plt.errorbar(x, y=gaal_aver, yerr=gaal_sd, color='k', capsize=2, label="GAAL")
    plt.errorbar(x, y=random_aver, yerr=random_sd, color='c', capsize=2, label="Random Sampling")
    full_aver = np.mean(full_list)  # single value
    plt.axhline(y=full_aver, color='r', linestyle=':', label="Fully Supervised")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(f"{result_dir}/all.png")
    plt.show()
    plt.close()

    for i in range(10):
        plt.plot(x, gaal_list[i], color='k', label="GAAL")
        plt.plot(x, random_list[i], color='c', label="Random Sampling")
        plt.axhline(y=full_aver, color='r', linestyle=':', label="Fully Supervised")
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(f"{result_dir}/{i+1}.png")
        # plt.show()
        plt.close()





