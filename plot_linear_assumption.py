import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

save_data = np.load('save_ppop_ppoL0.npz')

save_theta_rev_cost = save_data['theta']
save_context = save_data['context']
save_rev_cost_im = save_data['revcost']

rew_indx = 0
cost_indx = 1
est_rew = np.array([ np.dot(save_context[t,:], save_theta_rev_cost[-1,:, rew_indx]) \
                        for t in range(save_context.shape[0])])
est_cost = np.array([ np.dot(save_context[t,:], save_theta_rev_cost[-1,:, cost_indx]) \
                        for t in range(save_context.shape[0])])
plt.figure()
plt.plot(save_rev_cost_im[:1000,rew_indx], color='b', linewidth=3, label='Immediate reward')
plt.plot(est_rew[:1000], color='r', linewidth=3, label='Estimated reward')
plt.xlabel('Reward')
plt.ylabel('Environment interacts (t)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(save_rev_cost_im[:1000,cost_indx], color='b', linewidth=3, label='Immediate cost')
plt.plot(est_cost[:1000], color='r', linewidth=3, label='Estimate cost')
plt.xlabel('Cost')
plt.ylabel('Environment interacts (t)')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.show()
