import numpy as np
import random
import time
import tensorflow as tf
import joblib
import os
import os.path as osp
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.logx import restore_tf_graph, EpochLogger
from safe_rl.utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum
from optim_solver import compute_next_theta_hat, compute_estimate_reward

##### Load policy from a saved file --> Restore the tensorflow graph of learned policy ######
def load_policy(fpath, itr='last', deterministic=False):
    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr
    # load the things!
    sess = tf.Session(graph=tf.Graph())
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))
    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']
    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]
    get_last_hidden_layer_pi = lambda x: sess.run(model['b_pi'], feed_dict={model['x']: x[None, :]})[0]
    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None
    return env, get_action, get_last_hidden_layer_pi, sess

def gamma_next(Z, t, kappa, D, lam, delta, R=10):
    U = D
    return 16 * ((R+U)**2 / kappa) * np.log(2.0/delta * np.sqrt(1 + 4 * D**2 * t)) + lam * D**2 +\
           2 * ((R+U)**2 /kappa) * np.log(np.linalg.det(Z) / np.log(lam**(Z.shape[0]))) + kappa/2.0

def MOGLB_UCB(
    env, #Safe-gym environment
    get_action_safe,
    get_last_hl_safe,
    get_action_performant,
    get_last_hl_perf,
    lam = 1.0,
    kappa=1.0,
    epochs=50,
    seed=0,
    steps_per_epoch=4000,
    max_ep_len=1000,
    save_freq = 1,
    D=1,
    delta=0.1
    logger=None,
    logger_kwargs=dict()
):
    # Initialize the logger
    logger = EpochLogger(**logger_kwargs) if logger is None else logger
    local_dict = locals()
    del local_dict['env']
    logger.save_config(local_dict)
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    assert(lam >= max(1,kappa/2)), 'labmda should be greater than max(1,kappa/2)'
    ####### Get the size of the features ####################
    o = env.reset()
    last_hl_perf = get_last_hl_perf(o)
    last_hl_safe = get_last_hl_safe(o)
    d = len(last_hl_perf) + len(last_hl_safe) + 1
    # Initialization of Z_t, theta_hat and an approximation of the pareto front O_t
    theta_hat_r_t = np.zeros((d,1))
    theta_hat_c_t = np.zeros((d,1))
    Z_t = lam * np.identity(d)
    O_t = np.array([0,1]) # The Pareto set starts off from the set of all arms 0 is safe and 1 is performant 
    # Safety-gym environment interaction loop
    start_time = time.time()
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    cum_cost = 0

    step_counter = 0
    for epoch in range(epochs):
        # select a policy at random from the Pareto set
        policy_t = np.random.choice(O_t)
        a = get_action_safe(o) if policy_t == 0 else get_action_performant(o)
        # Get the reward and cost from the environment by step
        o2, r, done, info = env.step(a)
        c = info.get('cost', 0)
        # Build the feature vector
        last_hl_perf = np.array(get_last_hl_perf(o))
        last_hl_perf = last_hl_perf.reshape((last_hl_perf.shape[0],1))
        last_hl_safe = np.array(get_last_hl_safe(o))
        last_hl_safe = last_hl_safe.reshape((last_hl_safe.shape[0],1))
        x_t = np.concatenate((last_hl_safe, last_hl_perf, np.array([[policy_t]])), axis=0)
        # Observe reward
        y_t = np.array([r , -c])
        # Update the PSD matrix Z
        Z_t += 0.5* kappa * np.matmul(x_t , x_t.T)
        # Update theta_hat
        gradR_t = (-y_t[0] + np.matmul(theta_hat_r_t.T,x_t)[0,0]) * x_t
        gradC_t = (-y_t[1] + np.matmul(theta_hat_c_t.T,x_t)[0,0]) * x_t
        theta_hat_r_t = compute_next_theta_hat(theta_hat_r_t, D, Z_t, gradR_t)
        theta_hat_c_t = compute_next_theta_hat(theta_hat_c_t, D, Z_t, gradC_t)
        # Compute gamme_t
        gamma_t  = gamma_next(Z_t, step_counter, kappa, D, lam, delta)
        # compute UCB indices - step 11
        ucb_r_safe = compute_estimate_reward(theta_hat_reward,Z,gamma,get_feature('safe',o))
        ucb_r_perf = compute_estimate_reward(theta_hat_reward,x_t,Z,gamma,x_t,get_feature('performant',o))
        ucb_c_safe = compute_estimate_cost(theta_hat_cost,x_t,Z,gamma,get_feature('safe',o))
        ucb_c_perf = compute_estimate_cost(theta_hat_cost,x_t,Z,gamma,get_feature('performant',o))
        # Logger informations
        o = o2
        cum_cost += c # Track cumulative cost over training
        ep_ret += r
        ep_cost += c
        ep_len += 1
        step_counter += 1
    #     terminal = done or (ep_len == max_ep_len)
    #     if terminal:
    #         logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
    #         o, r, done, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
    #     else:
    #         print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)

    #     cumulative_cost = cum_cost
    #     cost_rate = cumulative_cost / ((epoch + 1) * steps_per_epoch)

    # #     # Compute the feature representation
    #     x_t = np.concatenate((get_action_safe(o),get_action_performant(o),np.array([1 if policy_t == 'performant' else 0])))
    #     temp = x_t.reshape((x_t.size,1))
    # #     # Update Z - step 5
    #     Z += (kappa / 2) * np.matmul(temp, temp.transpose())
    #     print(Z)
    # #
    #     y_t = np.array([r,-c])
    # #
    # #     # Update theta_hat - step 7
    #     theta_hat_reward = compute_next_theta_hat(theta_hat_reward, D, Z, -y_t[0] * x_t + np.dot(theta_hat_reward,x_t) * x_t)
    #     theta_hat_cost = compute_next_theta_hat(theta_hat_cost, D, Z, -y_t[1] * x_t + np.dot(theta_hat_cost,x_t) * x_t)
    #
    #     # Update gamma - step 8
    #     gamma = gamma_next(Z,t,D,l
    #
    #     # compute UCB indices - step 11
    #
    #     estimate_reward_safe = compute_estimate_reward(theta_hat_reward,Z,gamma,get_feature('safe',o))
    #     estimate_reward_performant = compute_estimate_reward(theta_hat_reward,x_t,Z,gamma,x_t,get_feature('performant',o))
    #     estimate_cost_safe = compute_estimate_cost(theta_hat_cost,x_t,Z,gamma,get_feature('safe',o))
    #     estimate_cost_performant = compute_estimate_cost(theta_hat_cost,x_t,Z,gamma,get_feature('performant',o))
    #
    #     # Update the Pareto optimal set
    #     if estimate_cost_performant >= estimate_cost_safe and estimate_reward_performant > estimate_reward_safe:
    #         Pareto_arms = ['performant']
    #     elif estimate_cost_performant > estimate_cost_safe and estimate_reward_performant >= estimate_reward_safe:
    #         Pareto_arms = ['performant']
    #     elif estimate_cost_performant <= estimate_cost_safe and estimate_reward_performant < estimate_reward_safe:
    #         Pareto_arms = ['safe']
    #     elif estimate_cost_performant < estimate_cost_safe and estimate_reward_performant <= estimate_reward_safe:
    #         Pareto_arms = ['safe']
    #     else:
    #         Pareto_arms = ['safe','performant']
    #
    #
    #     # =====================================================================#
    #     #  Log performance and stats                                          #
    #     # =====================================================================#
    #
    #     logger.log_tabular('Epoch', epoch)
    #
    #     # Performance stats
    #     logger.log_tabular('EpRet', with_min_and_max=True)
    #     logger.log_tabular('EpCost', with_min_and_max=True)
    #     logger.log_tabular('EpLen', average_only=True)
    #     logger.log_tabular('CumulativeCost', cumulative_cost)
    #     logger.log_tabular('CostRate', cost_rate)
    #
    #         # Value function values
    #     logger.log_tabular('VVals', with_min_and_max=True)
    #     logger.log_tabular('CostVVals', with_min_and_max=True)
    #
    #         # Pi loss and change
    #     logger.log_tabular('LossPi', average_only=True)
    #     logger.log_tabular('DeltaLossPi', average_only=True)
    #
    #         # Surr cost and change
    #     logger.log_tabular('SurrCost', average_only=True)
    #     logger.log_tabular('DeltaSurrCost', average_only=True)
    #
    #         # V loss and change
    #     logger.log_tabular('LossV', average_only=True)
    #     logger.log_tabular('DeltaLossV', average_only=True)
    #
    #         # Vc loss and change, if applicable (reward_penalized agents don't use vc)
    #     if not (agent.reward_penalized):
    #         logger.log_tabular('LossVC', average_only=True)
    #         logger.log_tabular('DeltaLossVC', average_only=True)
    #
    #     if agent.use_penalty or agent.save_penalty:
    #         logger.log_tabular('Penalty', average_only=True)
    #         logger.log_tabular('DeltaPenalty', average_only=True)
    #     else:
    #         logger.log_tabular('Penalty', 0)
    #         logger.log_tabular('DeltaPenalty', 0)
    #
    #         # Anything from the agent?
    #     # agent.log()
    #
    #         # Policy stats
    #     logger.log_tabular('Entropy', average_only=True)
    #     logger.log_tabular('KL', average_only=True)
    #
    #         # Time and steps elapsed
    #     logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
    #     logger.log_tabular('Time', time.time() - start_time)
    #
    #         # Show results!
    #     logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath_safe', type=str)
    parser.add_argument('fpath_performant', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--exp_name', type=str, default='blending_policy')
    args = parser.parse_args()
    _, get_action_safe, get_last_hl_pi_safe, sess_safe = load_policy(args.fpath_safe, args.itr if args.itr >= 0 else 'last', args.deterministic)
    env, get_action_perf, get_last_hl_pi_perf, sess_perf = load_policy(args.fpath_performant, args.itr if args.itr >= 0 else 'last', args.deterministic)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    MOGLB_UCB(env, get_action_safe, get_last_hl_pi_safe, get_action_perf, get_last_hl_pi_perf, logger_kwargs=logger_kwargs, seed=args.seed)