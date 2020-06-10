import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

def moving_average(a, n=3) :
    """Compute a moving average for smoothing the curves"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def read_data(filename, num_traces, dict_key):
    """Read the data from the saved file.
    The number of traces represents the number of parrallel
    run that has been done.
    """
    res_data =  dict()
    for i in range(num_traces):
        mData = np.load(filename+str(i)+'.npz')
        for key in dict_key:
            if key not in mData:
                continue
            if key not in res_data:
                res_data[key] = np.zeros([num_traces]+[mData[key].shape[k] \
                                            for k in range(mData[key].ndim)])
            # In case the traces does not have the same size we will consider
            # TThe lowest later
            if mData[key].ndim == 1:
                res_data[key][i][:mData[key].shape[0]] = mData[key][:]
            else:
                res_data[key][i][:mData[key].shape[0],:mData[key].shape[1]] = mData[key][:,:]
    return res_data

def preprocess_cost_reward(data, ind_trace, steps_per_epoch, window=1):
    """Preprocess the given cost and the given reward and return 
    the mean, mean+std and mean-std values as metrics. This function
    requires data from a single trace (execution). EVolution with respect
    to the number of epochs.
    """
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
    """Preprocess the cost rate from a single trace and return it evolution
    with respect to the number of epochs.
    """
    current_data = data[ind_trace]
    time_axis = np.array([ (i+1)*steps_per_epoch for i in range(current_data.shape[0])])
    if window > 2:
        time_axis = moving_average(time_axis, window)
        current_data = moving_average(current_data, window)
    return time_axis, current_data

def preprocess_pr(data_arm, data_dx, ind_trace, 
                    steps_per_epoch, max_ep_len):
    """Preprocess the ratio of corrected picked arm
    and return
    """
    time_axis = np.array([i for i in range(data_arm.shape[1])])
    res_dx = np.zeros((data_arm.shape[0],data_arm.shape[1]))
    for i in range(data_arm.shape[0]):
        for j in range(data_arm.shape[1]):
            res_dx[i,j] = data_dx[i, int(data_arm[i,j]), j]

    # FInd the time for which the step size pareto gap was 0
    res_dx = 1 - (res_dx[ind_trace,:] > 1e-5)
    mean_res = res_dx
    listRes = list()

    # Do an average of the arm correctly picked every max_ep_len
    for i in range(0,res_dx.shape[0], max_ep_len):
        listRes.append(np.mean(mean_res[i:(i+max_ep_len)]))

    # Perfdorm statistics over an epoch now
    listBis = np.array(listRes)
    finalMean = list()
    finalStd = list()
    for j in range(0, len(listRes), int(steps_per_epoch/max_ep_len)):
        # j+10
        finalMean.append(np.mean(listBis[j:(j+int(steps_per_epoch/max_ep_len))]))
        finalStd.append(np.std(listBis[j:(j+int(steps_per_epoch/max_ep_len))]))

    # Save the result
    time_axis = np.array([i*steps_per_epoch for i in range(len(finalMean))])
    finalMean = np.array(finalMean)
    finalMin = finalMean - np.array(finalStd)
    finalMax = finalMean + np.array(finalStd)
    return time_axis, finalMean, finalMin, finalMax

def parse_data(logdir, legend, colors, dict_key, num_traces,
                ind_traces, window=1, steps_per_epoch=30000,
                max_ep_len=1000):
    final_dict = dict()
    for path_data, legend_data, color_data, n_trace, ind_trace in \
                zip(logdir, legend, colors, num_traces, ind_traces):
         res_data = read_data(path_data, n_trace, dict_key)
         for key in res_data:
            if key == 'ap':
                continue
            if key not in final_dict:
            	final_dict[key] = dict()
         for key in res_data:
            if key in ['ret', 'cost']:
                final_dict[key][legend_data] = (color_data, \
                    preprocess_cost_reward(res_data[key], ind_trace,
                                            steps_per_epoch, window))
            if key in ['px']:
                final_dict[key][legend_data] = (color_data, \
                    preprocess_pr(res_data['ap'], res_data[key],ind_trace,
                                    steps_per_epoch, max_ep_len))
            if key in ['crate']:
                final_dict[key][legend_data] = (color_data, \
                    preprocess_cost_rate(res_data[key], ind_trace, steps_per_epoch, window))
    return final_dict

def plot_result(dict_key_y, dict_key_x, finalData, outputName, cutData=1):
    """Plot the resulting data
    """
    for key, dictController in finalData.items():
        plt.figure()
        for legend, (color, data) in dictController.items():
            plt.plot(data[0][:cutData], data[1][:cutData], color=color, alpha=1.0, label=legend)
            if len(data) > 2:
                plt.fill_between(data[0][:cutData], data[2][:cutData], data[3][:cutData], alpha=0.2,
                                facecolor=color, edgecolor=color, linewidth=3)
        if key == 'px':
            plt.ylim([0.4,1.0])
        plt.legend(loc='best')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.xlabel(dict_key_x[key])
        plt.ylabel(dict_key_y[key])
        plt.grid(True)
        plt.tight_layout()
        tikzplotlib.save(outputName + key + '.tex')
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
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--max_epochs_plot', type=int, default=100)
    parser.add_argument('--ind_traces', nargs='+', type=int)
    parser.add_argument('--output_name',type=str)
    args = parser.parse_args()
    if args.ind_traces is None:
        args.ind_traces = [0 for i in args.legend]
    if args.num_traces is None:
    	args.num_traces = [1 for i in args.legend]

    # Axis label
    dict_key_y = {'ap' : 'Arms picked', 'px' : 'Ratio of correct arms',
                'ret' : 'Average reward', 'cost' : 'Average cost',
                'crate' : 'Cost rate'}
    dict_key_x = {'ap' : 'Arms picked', 'px' : '$T$',
                'ret' : '$T$', 'cost' : '$T$',
                'crate' : '$T$'}

    # Parse and obtain the final data
    finalData = parse_data(args.logdir, args.legend, args.colors, dict_key_y,
                    args.num_traces, args.ind_traces, args.window,
                    args.steps_per_epoch, args.max_ep_len)

    # Plot the data and save them
    plot_result(dict_key_y, dict_key_x, finalData, args.output_name, cutData=args.max_epochs_plot)