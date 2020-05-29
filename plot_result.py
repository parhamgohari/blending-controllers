import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import tikzplotlib

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def read_data(filename, num_traces, dict_key):
    res_data =  dict()
    for i in range(num_traces):
        mData = np.load(filename+str(i)+'.npz')
        for key in dict_key:
            if key not in mData:
                continue
            if key not in res_data:
                res_data[key] = np.zeros([num_traces]+[mData[key].shape[k] \
                                            for k in range(mData[key].ndim)])
            # print (res_data[key].shape)
            res_data[key][i] = mData[key]
    return res_data

def preprocess_cost_reward(data, ind_trace, steps_per_epoch, window=1):
    current_data = data[ind_trace]
    time_axis = np.array([ (i+1)* steps_per_epoch for i in range(current_data.shape[0])])
    std_val = np.std(current_data, axis=1)
    mean_val = np.mean(current_data, axis=1)
    val_min = mean_val-std_val
    val_max = mean_val+std_val
    if window > 2:
        time_axis = moving_average(time_axis, window)
        mean_val = moving_average(mean_val, window)
        val_min = moving_average(val_min, window)
        val_max = moving_average(val_max, window)
    return time_axis, mean_val, val_min, val_max

def preprocess_cost_rate(data, ind_trace, steps_per_epoch, window=1):
    current_data = data[ind_trace]
    print (current_data)
    time_axis = np.array([ (i+1)*steps_per_epoch for i in range(current_data.shape[0])])
    if window > 2:
        time_axis = moving_average(time_axis, window)
        current_data = moving_average(current_data, window)
    return time_axis, current_data

def preprocess_pr(data_arm, data_dx, window=1):
    time_axis = np.array([i for i in range(data_arm.shape[1])])
    res_dx = np.zeros((data_arm.shape[0],data_arm.shape[1]))
    for i in range(data_arm.shape[0]):
        for j in range(data_arm.shape[1]):
            res_dx[i,j] = data_dx[i, int(data_arm[i,j]), j]
    std_res = np.std(res_dx, axis=0)
    mean_res = np.mean(res_dx, axis=0)
    min_val = mean_res - std_res
    max_val = mean_res + std_res
    if window > 2:
        mean_res = moving_average(mean_res, window)
        min_val = moving_average(min_val, window)
        max_val = moving_average(max_val, window)
        time_axis = moving_average(time_axis, window)
    return time_axis, mean_res, min_val, max_val

def parse_data(logdir, legend, colors, dict_key, num_traces,
                ind_traces, window=1, steps_per_epoch=30000):
    final_dict = dict()
    for path_data, legend_data, color_data, n_trace, ind_trace in \
                zip(logdir, legend, colors, num_traces, ind_traces):
         res_data = read_data(path_data, n_trace, dict_key)
         for key in res_data:
            if key == 'ap':
                continue
            final_dict[key] = dict()
         for key in res_data:
            if key in ['ret', 'cost']:
                final_dict[key][legend_data] = (color_data, \
                    preprocess_cost_reward(res_data[key], ind_trace,
                                            steps_per_epoch, window))
            if key in ['px']:
                final_dict[key][legend_data] = (color_data, \
                    preprocess_pr(res_data['ap'], res_data[key], window))
            if key in ['crate']:
                final_dict[key][legend_data] = (color_data, \
                    preprocess_cost_rate(res_data[key], ind_trace, steps_per_epoch, window))
    return final_dict



def plot_result(dict_key_y, dict_key_x, finalData):
    for key, dictController in finalData.items():
        plt.figure()
        for legend, (color, data) in dictController.items():
            plt.plot(data[0], data[1], color=color, alpha=1.0, label=legend)
            if len(data) > 2:
                plt.fill_between(data[0], data[2], data[3], alpha=0.2,
                                facecolor=color, edgecolor=color, linewidth=3)
        plt.legend(loc='best')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.xlabel(dict_key_x[key])
        plt.ylabel(dict_key_y[key])
        plt.grid(True)
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', nargs='+', required=True)
    parser.add_argument('--legend', nargs='+', required=True)
    parser.add_argument('--colors', nargs='+', required=True)
    parser.add_argument('--num_traces', nargs='+', required=True, type=int)
    parser.add_argument('--window', '-w', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=30000)
    parser.add_argument('--ind_traces', nargs='+', type=int)
    args = parser.parse_args()
    if args.ind_traces is None:
        args.ind_traces = [0 for i in args.legend]
    # The yAxis of the different plots
    dict_key_y = {'ap' : 'Arms picked', 'px' : '$\Delta x_t$',
                'ret' : 'Episode reward', 'cost' : 'Episode cost',
                'crate' : 'Cost rate'}
    dict_key_x = {'ap' : 'Arms picked', 'px' : 'Environment Interacts',
                'ret' : 'Environment Interacts', 'cost' : 'Environment Interact',
                'crate' : 'Environment Interact'}

    finalData = parse_data(args.logdir, args.legend, args.colors, dict_key_y,
                    args.num_traces, args.ind_traces, args.window,
                    args.steps_per_epoch)
    plot_result(dict_key_y, dict_key_x, finalData)
    # print (finalData)
