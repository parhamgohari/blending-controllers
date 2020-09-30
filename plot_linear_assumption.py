import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

def moving_average(a, n=3) :
    """Compute a moving average for smoothing the curves"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

save_data = np.load('blend_point_goal1_trpo_test0.npz')

save_theta_rev_cost = save_data['theta']
save_context = save_data['context']
save_rev_cost_im = save_data['revcost']

rew_indx = 0
cost_indx = 1

last_index = save_context.shape[0]

norm_theta_rew = np.array([ np.linalg.norm(save_theta_rev_cost[t,:, rew_indx] - np.array([1.0,0])) for t in range(1,last_index)])
norm_theta_cost = np.array([ np.linalg.norm(save_theta_rev_cost[t,:, cost_indx] - np.array([0,1.0])) for t in range(1,last_index)])

curBis = 25000
plt.figure()
plt.plot(norm_theta_rew[:curBis], color='b', linewidth=3, label=r'$||\theta_{*,\mathrm{reward}}- \hat{\theta}_{t,\mathrm{reward}}||_2$')
plt.plot(norm_theta_cost[:curBis], color='g', linewidth=3, label=r'$||\theta_{*,\mathrm{cost}}- \hat{\theta}_{t,\mathrm{cost}}||_2$')
# plt.plot(time_step, est_rew, color='r', linewidth=3, label='Estimated reward')
plt.xlabel(r'Environment interacts')
plt.ylabel(r'$||\theta_{*,}- \hat{\theta}_{t,}||_2$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('difftheta.png', transparent=True)
tikzplotlib.save('difftheta.tex')

plt.show()
exit()
est_rew = np.array([ np.dot(save_context[t,:], save_theta_rev_cost[-1,:, rew_indx]) \
                        for t in range(1,last_index)])
est_cost = np.array([ np.dot(save_context[t,:], save_theta_rev_cost[-1,:, cost_indx]) \
                        for t in range(1,last_index)])

# theta_id = np.array([[1.0,0],[0,1.0]])
est_rew_id = np.array([ np.dot(save_context[t,:], np.array([1.0,0])) \
                        for t in range(1,last_index)])
est_cost_id = np.array([ np.dot(save_context[t,:], np.array([0,1.0])) \
                        for t in range(1,last_index)])

mean_val_r = np.mean(est_rew - est_rew_id)
std_dev_r = np.std(est_rew - est_rew_id)

mean_val_c = np.mean(est_cost - est_cost_id)
std_dev_c = np.std(est_cost - est_cost_id)

print ('Mean rew : {}, std dev rew : {}'.format(mean_val_r, std_dev_r))
print ('Mean cost : {}, std dev cost : {}'.format(mean_val_c, std_dev_c))

# exit()

time_step = [ i for i in range(1,last_index)]
window_avg = 1
cut = range(3296700,3297700)
est_rew = moving_average(est_rew, window_avg)[cut]
est_cost = moving_average(est_cost, window_avg)[cut]
im_rew = moving_average(save_rev_cost_im[1:last_index,rew_indx], window_avg)[cut]
im_cost = moving_average(save_rev_cost_im[1:last_index,cost_indx], window_avg)[cut]
time_step = moving_average(time_step, window_avg)[cut]
est_rew_id = moving_average(est_rew_id, window_avg)[cut]
est_cost_id = moving_average(est_cost_id, window_avg)[cut]

mean_val_r = np.mean(est_rew - est_rew_id)
std_dev_r = np.std(est_rew - est_rew_id)

mean_val_c = np.mean(est_cost - est_cost_id)
std_dev_c = np.std(est_cost - est_cost_id)

time_step = np.array([i for i in range(time_step.shape[0])])

plt.figure()
plt.plot(time_step, im_rew, color='b', linewidth=3, label='Immediate reward')
plt.plot(time_step, est_rew_id, color='g', linewidth=3, label='Ideal reward')
plt.plot(time_step, np.abs(est_rew_id - im_rew), linewidth=3)
# plt.plot(time_step, est_rew, color='r', linewidth=3, label='Estimated reward')
plt.xlabel('Environment interacts (t)')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
tikzplotlib.save('rew.tex')

plt.figure()
plt.scatter(time_step, est_rew_id - im_rew, color='m', marker='s', s=4)
plt.plot(time_step, [ mean_val_r for i in range(time_step.shape[0])], color='g', linewidth=3)
# plt.fill_between(time_step, [ mean_val_r - 3*std_dev_r for i in range(time_step.shape[0])], [ mean_val_r + 3*std_dev_r for i in range(time_step.shape[0])], 
# 					facecolor='grey', edgecolor='darkgrey', linewidth=3)
# plt.plot(time_step, est_rew, color='r', linewidth=3, label='Estimated reward')
plt.xlabel('Environment interacts (t)')
plt.ylabel('Difference Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
tikzplotlib.save('diff_rew.tex')

plt.figure()
plt.plot(time_step, im_cost, color='b', linewidth=3, label='Immediate cost')
plt.plot(time_step, est_cost_id, color='g', linewidth=3, label='Ideal reward')
# plt.plot(time_step, est_cost, color='r', linewidth=3, label='Estimate cost')
plt.xlabel('Cost')
plt.ylabel('Environment interacts (t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
tikzplotlib.save('cost.tex')

plt.figure()
plt.scatter(time_step, est_cost_id - im_cost, color='m', marker='s', s=4)
plt.plot(time_step, [ mean_val_c for i in range(time_step.shape[0])], color='g', linewidth=3)
# plt.fill_between(time_step, [ mean_val_c - 3*std_dev_c for i in range(time_step.shape[0])], [ mean_val_c + 3*std_dev_c for i in range(time_step.shape[0])], 
# 					facecolor='grey', edgecolor='darkgrey', linewidth=3)
# plt.plot(time_step, est_rew, color='r', linewidth=3, label='Estimated reward')
plt.xlabel('Environment interacts (t)')
plt.ylabel('Difference cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
tikzplotlib.save('diff_cost.tex')

# plt.figure()
# plt.plot(time_step, save_theta_rev_cost[-1,:, rew_indx], color='b', linewidth=3, label='Theta reward')
# plt.xlabel('Environment interacts (t)')
# plt.ylabel('Reward')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# plt.figure()
# plt.plot(time_step, save_theta_rev_cost[-1,:, cost_indx], color='b', linewidth=3, label='Theta reward')
# plt.xlabel('Environment interacts (t)')
# plt.ylabel('Reward')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()


plt.show()
