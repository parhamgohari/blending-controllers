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
from optim_solver import compute_next_theta_hat, compute_estimate_reward, compute_estimate_reward_fast

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
    get_v = lambda x: sess.run(model['v'], feed_dict={model['x']: x[None, :]})[0]
    get_vc = lambda x: sess.run(model['vc'], feed_dict={model['x']: x[None, :]})[0]
    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None
    return env, get_action, get_last_hidden_layer_pi, get_v, get_vc, sess

def gamma_next(Z, t, kappa, D, lam, delta, R=1.0):
    U = D
    s, val = np.linalg.slogdet(Z)
    val = s * val
    # print (np.linalg.det(Z))
    return 16 * ((R+U)**2 / kappa) * np.log(2.0/delta * np.sqrt(1 + 4 * D**2 * t)) + lam * D**2 +\
           2 * ((R+U)**2 /kappa) * val + kappa/2.0

def MOGLB_UCB(
    env, #Safe-gym environment
    get_action_safe,
    get_last_hl_safe,
    get_action_performant,
    get_last_hl_perf,
    lam = 1.0,
    kappa=0.1,
    num_env_interact=1000000,
    steps_per_epoch = 30000,
    seed=0,
    max_ep_len=1000,
    save_freq = 50,
    D=1.0,
    delta=0.1,
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
    # Save the different mu and action for computing the pareto regret
    mu_act = np.zeros((num_env_interact,5))
    ####### Get the size of the features ####################
    o = env.reset()
    # last_hl_perf = get_last_hl_perf(o)
    # last_hl_safe = get_last_hl_safe(o)
    # last_hl_perf = get_action_performant(o)
    # last_hl_safe = get_action_safe(o)
    # print (get_last_hl_safe[0](o), get_last_hl_safe[1](o))
    # rand_ind_perf = np.random.choice(range(len(last_hl_perf)),10, replace=False)
    # rand_ind_safe = np.random.choice(range(len(last_hl_safe)),10, replace=False)
    # d = len(last_hl_perf) + len(last_hl_safe) + 1
    #  d = rand_ind_safe.shape[0] + rand_ind_perf.shape[0] + 2
    # d = len(last_hl_perf) + len(last_hl_perf)
    d =  2
    # Initialization of Z_t, theta_hat and an approximation of the pareto front O_t
    theta_hat_r_t = np.zeros((d,1))
    theta_hat_c_t = np.zeros((d,1))
    Z_t = lam * np.identity(d)
    O_t = np.array([0,1]) # The Pareto set starts off from the set of all arms 0 is safe and 1 is performant 
    # Safety-gym environment interaction loop
    start_time = time.time()
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    cum_cost = 0
    epochs = int(num_env_interact/steps_per_epoch)

    for t in range(num_env_interact):
        # print (t)
        # print (O_t)
        # select a policy at random from the Pareto set
        policy_t = np.random.choice(O_t)
        a = get_action_safe(o) if policy_t == 0 else get_action_performant(o)
        # bit_flip = np.array([[0],[0]])
        # bit_flip[policy_t,0] = 1
        # Get the reward and cost from the environment by step
        o2, r, done, info = env.step(a)
        # env.render()
        c = info.get('cost', 0)
        # Build the feature vector
        # last_hl_perf = np.array(get_last_hl_perf(o))
        # last_hl_perf = np.array(get_last_hl_perf(o))[rand_ind_perf]
        # last_hl_perf = last_hl_perf.reshape((last_hl_perf.shape[0],1))
        # # last_hl_safe = np.array(get_last_hl_safe(o))
        # last_hl_safe = np.array(get_last_hl_safe(o))[rand_ind_safe]
        # last_hl_safe = last_hl_safe.reshape((last_hl_safe.shape[0],1))
        # x_t = np.concatenate((last_hl_safe, last_hl_perf, bit_flip), axis=0)
        # x_t = np.array(a).reshape((len(a),1))
        x_t_s = np.array([get_last_hl_safe[0](o), get_last_hl_safe[1](o)]).reshape((d,1))
        norm_s = np.linalg.norm(x_t_s)
        x_t_p = np.array([get_last_hl_perf[1](o), get_last_hl_perf[0](o)]).reshape((d,1))
        norm_p = np.linalg.norm(x_t_p)
        if norm_s > 1e-12:
            x_t_s *= 1.0/ norm_s
        if norm_p > 1e-12:
            x_t_p *= 1.0/norm_p

        if policy_t == 0:
            x_t = x_t_s
        else:
            x_t = x_t_p
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
        gamma_t  = gamma_next(Z_t, t, kappa, D, lam, delta)
        # print (gamma_t)
        # compute UCB indices - step 11
        # x_t[d-2,0] = 1
        # x_t[d-1,0] = 0
        # if policy_t == 0:
        #     x_t_s = x_t
        #     x_t_p = x_t_o
        # else:
        #     x_t_s = x_t_o
        #     x_t_n = x_t
        inv_Z_x = np.linalg.solve(Z_t , x_t_s)
        ucb_r_safe = compute_estimate_reward_fast(theta_hat_r_t,inv_Z_x,gamma_t,x_t_s)
        ucb_c_safe = compute_estimate_reward_fast(theta_hat_c_t,inv_Z_x,gamma_t,x_t_s)
        # x_t[d-2,0] = 0
        # x_t[d-1,0] = 1
        inv_Z_x = np.linalg.solve(Z_t , x_t_p)
        ucb_r_perf = compute_estimate_reward_fast(theta_hat_r_t,inv_Z_x,gamma_t,x_t_p)
        ucb_c_perf = compute_estimate_reward_fast(theta_hat_c_t,inv_Z_x,gamma_t,x_t_p)
        mu_act[t,:] = np.array([ucb_r_safe,ucb_c_safe,ucb_r_perf,ucb_c_perf,policy_t])
        # print (ucb_r_safe,ucb_c_safe,ucb_r_perf,ucb_c_perf)
        # print (np.matmul(theta_hat_r_t.T , x_t))
        # Update the pareto optimal set
        if ucb_c_perf >= ucb_c_safe and ucb_r_perf > ucb_r_safe: # The performant policy Pareto dominates the safe policy
            O_t = np.array([1]) # The approximated Pareto front contains the performant policy
        elif ucb_c_perf > ucb_c_safe and ucb_r_perf >= ucb_r_safe:
            O_t = np.array([1])
        elif ucb_c_perf <= ucb_c_safe and ucb_r_perf < ucb_r_safe: # The safe policy Pareto dominates the performnat policy
            O_t = np.array([0]) # The approximated Pareto front contains the safe policy
        elif ucb_c_perf < ucb_c_safe and ucb_r_perf <= ucb_r_safe:
            O_t = np.array([0])
        else:
            O_t = np.array([0,1])
        # Logger informations
        o = o2
        cum_cost += c # Track cumulative cost over training
        ep_ret += r
        ep_cost += c
        ep_len += 1
        terminal = done or (ep_len == max_ep_len)
        if terminal:
            logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
            o, r, done, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
        # else:
        #     print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
        if t % 4000== 0:
            print (t / 4000 , mu_act[t,:] , gamma_t)
            np.save('theta_hat_r.npy', theta_hat_r_t)
            np.save('theta_hat_c.npy', theta_hat_c_t)
            np.save('pareto_regret.npy', mu_act)


        if t== 0 or t % steps_per_epoch != 0:
            continue
        
        # Save model
        epoch = int(t/steps_per_epoch)
        if ( epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        cost_rate = cum_cost / ((epoch+1) * steps_per_epoch)
    
        # =====================================================================#
        #  Log performance and stats                                          #
        # =====================================================================#
        logger.log_tabular('Epoch', epoch)
        # Performance stats
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cum_cost)
        logger.log_tabular('CostRate', cost_rate)
    
            # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', epoch * steps_per_epoch)
        logger.log_tabular('Time', time.time() - start_time)

        # Save theta_hat value
        np.save('theta_hat_r.npy', theta_hat_r_t)
        np.save('theta_hat_c.npy', theta_hat_c_t)
        np.save('pareto_regret.npy', mu_act)
        # Show results!
        logger.dump_tabular()

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
    _, get_action_safe, get_last_hl_pi_safe, get_v_safe, get_vc_safe, sess_safe = load_policy(args.fpath_safe, args.itr if args.itr >= 0 else 'last', args.deterministic)
    env, get_action_perf, get_last_hl_pi_perf, get_v_perf, get_vc_perf, sess_perf = load_policy(args.fpath_performant, args.itr if args.itr >= 0 else 'last', args.deterministic)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    # MOGLB_UCB(env, get_action_safe, get_last_hl_pi_safe, get_action_perf, get_last_hl_pi_perf, logger_kwargs=logger_kwargs, seed=args.seed)
    MOGLB_UCB(env, get_action_safe, [get_v_safe,get_vc_safe], get_action_perf, [get_v_perf,get_vc_perf], logger_kwargs=logger_kwargs, seed=args.seed)