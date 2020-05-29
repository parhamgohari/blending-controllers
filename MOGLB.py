import numpy as np
from pickle import dumps,loads
import time
import tensorflow as tf
import joblib
import os
import os.path as osp
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.logx import restore_tf_graph, EpochLogger
from safe_rl.utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum
from copy import deepcopy

def load_policy(fpath, itr='last', deterministic=False):
    '''
    Load policy from a saved file and restore the tensorflow graph of the
    actual policy
    @fpath          :   provides the path of the  policy of interest
    @deterministic  :   default false, provides if action selection should be
                        deterministic (better for SAC policies)
    @return         :   @env                : return the evironment
                        @get_action         : the state action funcrion
                        @get_objective      : get the reward and cost functions
                        @sess               : get the current sess
    '''
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
    get_v = lambda x: sess.run(model['v'], feed_dict={model['x']: x[None, :]})[0]
    get_vc = lambda x: sess.run(model['vc'], feed_dict={model['x']: x[None, :]})[0]
    get_objective = lambda x: np.array([[get_v(x)],[get_vc(x)]])
    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None
    return env, get_action, get_objective, sess

def feat_map(zt=None, xt= None, args_list=None, env=None):
    '''
    Compute the features Xt = \psi (z_t, x_t)
    @zt             :   current state of the environment
    @xt             :   arm that has been picked
    @args_list      :   (function that provides the action to perform given
                        state, [estimator for the objectives]) for all arm
    @env            :   the safegym environment
    @return         :   @(o2_true, r_true, done_true, info_true) : env infos
                        @yt     :   vector of reward and cost if others actions
                                    were picked, column i gives [r,c] for arm i
                        @feat_t :   features function given the current state
                                    column i gives features for arm i
    '''
    list_act = []
    feat_t = None
    for elem, (curr_act, objective_fun) in enumerate(args_list):
        if feat_t is None:
            featVal = objective_fun(zt)
            feat_t = featVal / np.linalg.norm(featVal)
        else:
            featVal = objective_fun(zt)
            feat_t = np.concatenate((feat_t, featVal/np.linalg.norm(featVal)),
                                        axis=1)
        list_act.append(curr_act(zt))
    o2_true, r_true, done_true, info_true, yt = env.step_simulate(xt, list_act)
    return (o2_true, r_true, done_true, info_true), yt, feat_t


def compute_pgap(objective):
    '''
    Compute the gap given a matrix whose column represent
    each arms and the row are the different objectives
    @Return     :   an array containing the pareto gap for each arm
    '''
    res = np.zeros(objective.shape[1])
    for x in range(res.shape[0]):
        temp_x = objective - np.tile(objective[:,x:(x+1)],2)
        res[x] = np.max(np.max(temp_x, axis=1))
    return res

def compute_pareto_nd(xt, objective):
    '''
    Compute the pareto non-dominace gap. Useful to show the
    subregret bounds
    '''
    eps = np.inf
    for j in range(objective.shape[0]):
        eps_j = -np.inf
        for x in range(objective.shape[1]):
            eps_temp = np.maximum(0,objective[j,x]-objective[j,xt])
            eps_j =  np.maximum(eps_j, eps_temp)
        # print (eps_j)
        eps = np.minimum(eps, eps_j)
    return eps

def blending_algorithm(env, feat_objective, lam=1.0, delta=0.1, sigma=1.0,
                       theta_max= 10, feat_max=1.0, num_env_interact=1e6,
                       steps_per_epoch=30000, seed=201, max_ep_len=1000,
                       save_freq=50, logger=None, logger_kwargs=dict(),
                       render=False, test=False, data_path=''):
    '''
        Blend multiple controllers that implement multiple objectives
    '''
    # Initialize the logger
    logger = EpochLogger(**logger_kwargs) if logger is None else logger
    local_dict = locals()
    del local_dict['env']
    del local_dict['feat_objective']
    logger.save_config(local_dict)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Safety-gym environment interaction loop
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    cum_cost = 0
    epochs = int(num_env_interact/steps_per_epoch)

    # Get the dimension of the problem
    nArm = len(feat_objective)
    d = feat_objective[0][1](o).shape[0]
    nObj = d

    # Theorem 2 works for lam >= max {1, feat_max**2}
    assert(lam >= max(1,feat_max**2)), \
                'labmda should be greater than max(1,L^2)'

    # Define the ellipsis radius function
    def radEllipsis(t):
        return sigma * np.sqrt(d * np.log((1+t*(feat_max**2/lam))/delta)) +\
                np.sqrt(lam) * theta_max

    # Initialization step of V, theta_hat and pareto front
    if not test:
        Vt = lam * np.identity(d)
        Xpast = np.zeros((d,nObj))
        t_init = 0
    else:
        savePolicy = np.load(data_path)
        Vt = savePolicy['Vt']
        Xpast = savePolicy['Xpast']
        t_init = savePolicy['t']

    theta_t = np.zeros((d,nObj))
    mu_t = np.zeros((nObj,nArm))
    Ot = np.array([i for i in range(nArm)])
    eps_hat = np.zeros(nArm)
    eps_true = np.zeros(nArm)
    cum_regret_true = 0
    cum_regret_approx = 0
    sum_pareto_regret = 0

    # Save performance time
    start_time = time.time()

    # Start the iteration
    for t in range(t_init, t_init+num_env_interact):
        if render:
            env.render()
            time.sleep(1e-3)
        # print (t, Ot)
        # select a policy at random from the Pareto set
        xt = np.random.choice(Ot)
        (o2, r, done, info), yt, Xt = feat_map(o, xt, feat_objective, env)
        # print(yt)
        # print ('----')
        # print(Xt)
        c = info.get('cost', 0)
        # Temp variable saving X1:t * Y1:t
        for i in range(nObj):
            Xpast[:,i] += Xt[:,xt]*yt[i,xt]
        # Compute Vt+1
        XtXtT = np.matmul(Xt[:,xt:(xt+1)], Xt[:,xt:(xt+1)].T)
        Vt += XtXtT
        # Diagonalize Vt since it's going to be used twice for theta_t+1 and mu
        eVal, P = np.linalg.eig(Vt)
        # print (eVal)
        VtInv = np.matmul(P, np.matmul(np.diag(1.0/eVal), P.T))
        VtSqrtInv = np.matmul(P, np.matmul(np.diag(1.0/np.sqrt(eVal)),P.T))
        # Compute theta_t+1
        for i in range(nObj):
            theta_t[:,i] = np.matmul(VtInv, Xpast[:,i:(i+1)]).flatten()
            # print (np.dot(theta_t[:,i], Xt[:,xt]), yt[i,xt])
        # Compute the radius of the ellipsis
        rE =  radEllipsis(t)
        # print (rE)
        # Compute the UCB index for each controllers
        for x in range(nArm):
            featX = Xt[:,x:(x+1)]
            n1 = np.linalg.norm(np.matmul(VtSqrtInv,featX))
            tempC = rE * (1.0/n1) * np.matmul(VtInv, featX)
            for i in range(nObj):
                mu_t[i,x] = np.dot(theta_t[:,i] + tempC[:,0], featX[:,0])
        # Compute the pareto gaps
        eps_hat = compute_pgap(mu_t)
        eps_true = compute_pgap(yt)
        Ot = np.argwhere(eps_hat == np.min(eps_hat)).flatten()
        cum_regret_approx += eps_hat[xt]
        cum_regret_true += eps_true[xt]
        sum_pareto_regret = (t*sum_pareto_regret + compute_pareto_nd(xt, yt))/(t+1)
        # print (yt)
        # print (eps_hat, eps_true)
        # print('----------------------------')

        # Logger informations
        o = o2
        cum_cost += c # Track cumulative cost over training
        ep_ret += r
        ep_cost += c
        ep_len += 1
        terminal = done or (ep_len == max_ep_len)
        if terminal:
            logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost,
                            ParetoRegret=sum_pareto_regret)
            if test:
                print('EpRet %.3f \t EpCost %.3f \t EpLen %d'%(ep_ret, ep_cost, ep_len))
            o, r, done, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0

        if test or t== 0 or t % steps_per_epoch != 0:
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
        logger.log_tabular('ParetoRegret', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CumulativeCost', cum_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('CumulativeRegretTrue', cum_regret_true)
        logger.log_tabular('CumulativeRegretApprox', cum_regret_approx)
        logger.log_tabular('DiffRegret', cum_regret_true-cum_regret_approx)
        logger.log_tabular('ParetoRegretSum', sum_pareto_regret)

        # Time and steps elapsed
        logger.log_tabular('TotalEnvInteracts', epoch * steps_per_epoch)
        logger.log_tabular('Time', time.time() - start_time)

        # Save learning values
        np.savez(data_path, Vt=Vt, Xpast=Xpast, t=t)

        # Show results!
        logger.dump_tabular()
    if test:
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.dump_tabular()

def blending_alg_np(env, feat_objective, lam=1.0, delta=0.05, sigma=1.0,
                       theta_max= 10, feat_max=1.0, num_run=10000,
                       max_ep_len=1000, max_env_interact_per_run=30000,
                       num_outer_run=10, save_data_path=''):
    '''
        Blend multiple controllers that implement multiple objectives
    '''
    # Get the dimension of the problem
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    nArm = len(feat_objective)
    d = feat_objective[0][1](o).shape[0]
    nObj = d
    # Theorem 2 works for lam >= max {1, feat_max**2}
    assert(lam >= max(1,feat_max**2)), \
                'labmda should be greater than max(1,L^2)'
    def radEllipsis(t):
        return sigma * np.sqrt(d * np.log((1+t*(feat_max**2/lam))/delta)) +\
                np.sqrt(lam) * theta_max
    # Data to be saved
    # reward_perf = np.zeros((num_run, max_env_interact_per_run))
    # reward_safe = np.zeros((num_run, max_env_interact_per_run))
    # cost_perf = np.zeros((num_run, max_env_interact_per_run))
    # cost_safe = np.zeros((num_run, max_env_interact_per_run))
    # reward_blend = np.zeros((num_run, max_env_interact_per_run))
    # cost_blend = np.zeros((num_run, max_env_interact_per_run))
    res_correct_arm = None
    res_avg_pr = None
    for k in range(num_outer_run):
        correct_arm_save = np.zeros((num_run,max_env_interact_per_run))
        avg_pr_regret_save=np.zeros((num_run,max_env_interact_per_run))
        # pr_regret_save=np.zeros((num_run,max_env_interact_per_run))
        for env_run in range(num_run):
            # Safety-gym environment interaction loop
            o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
            cum_cost = 0
            Vt = lam * np.identity(d)
            Xpast = np.zeros((d,nObj))

            theta_t = np.zeros((d,nObj))
            mu_t = np.zeros((nObj,nArm))
            Ot = np.array([i for i in range(nArm)])
            eps_hat = np.zeros(nArm)
            eps_true = np.zeros(nArm)
            cum_regret_true = 0
            cum_regret_approx = 0
            avg_sum_pareto_regret = 0
            sum_pareto_regret = 0
            # correct_arm = 0

            for t in range(max_env_interact_per_run):
                # select a policy at random from the Pareto set
                xt = np.random.choice(Ot)
                (o2, r, done, info), yt, Xt = feat_map(o, xt, feat_objective, env)
                c = info.get('cost', 0)
                # Temp variable saving X1:t * Y1:t
                for i in range(nObj):
                    Xpast[:,i] += Xt[:,xt]*yt[i,xt]
                # Compute Vt+1
                XtXtT = np.matmul(Xt[:,xt:(xt+1)], Xt[:,xt:(xt+1)].T)
                Vt += XtXtT
                # Diagonalize Vt since it's going to be used twice for theta_t+1 and mu
                eVal, P = np.linalg.eig(Vt)
                VtInv = np.matmul(P, np.matmul(np.diag(1.0/eVal), P.T))
                VtSqrtInv = np.matmul(P, np.matmul(np.diag(1.0/np.sqrt(eVal)),P.T))
                # Compute theta_t+1
                for i in range(nObj):
                    theta_t[:,i] = np.matmul(VtInv, Xpast[:,i:(i+1)]).flatten()
                # print(theta_t)
                # Compute the radius of the ellipsis
                rE =  radEllipsis(t)
                # Compute the UCB index for each controllers
                for x in range(nArm):
                    featX = Xt[:,x:(x+1)]
                    n1 = np.linalg.norm(np.matmul(VtSqrtInv,featX))
                    tempC = rE * (1.0/n1) * np.matmul(VtInv, featX)
                    for i in range(nObj):
                        mu_t[i,x] = np.dot(theta_t[:,i] + tempC[:,0], featX[:,0])
                # Compute the pareto gaps
                eps_hat = compute_pgap(mu_t)
                eps_true = compute_pgap(yt)
                Ot = np.argwhere(eps_hat == np.min(eps_hat)).flatten()
                cum_regret_approx += eps_hat[xt]
                cum_regret_true += eps_true[xt]
                temp_pr = compute_pareto_nd(xt, yt)
                avg_sum_pareto_regret = (t*avg_sum_pareto_regret + temp_pr)/(t+1)
                sum_pareto_regret += temp_pr
                # correct_arm += 1 if np.argmax(eps_true) in Ot else 0
                true_Ot = np.argwhere(eps_true == np.min(eps_true)).flatten()
                correct_arm = 1 if xt in true_Ot else 0
                # print(eps_true)
                # print(eps_hat)
                # print (yt)
                # print(mu_t)
                # print(xt, correct_arm)
                # print('--------------')

                # Save the information for plots
                # reward_perf[env_run, t] = yt[0,0]
                # reward_safe[env_run, t] = yt[0,1]
                # cost_perf[env_run, t] = yt[1,0]
                # cost_safe[env_run, t] = yt[1,1]
                # reward_blend[env_run, t] = r
                # cost_blend[env_run, t] = c
                correct_arm_save[env_run, t] = correct_arm
                avg_pr_regret_save[env_run, t] = avg_sum_pareto_regret
                # pr_regret_save[env_run, t] = sum_pareto_regret

                # Logger informations
                o = o2
                cum_cost += c # Track cumulative cost over training
                ep_ret += r
                ep_cost += c
                ep_len += 1
                terminal = done or (ep_len == max_ep_len)
                if terminal:
                    o, r, done, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
            # print(env_run, correct_arm, avg_sum_pareto_regret)
        new_correct_arm = np.mean(correct_arm_save, axis=0)
        if res_correct_arm is None:
            res_correct_arm = new_correct_arm.reshape(1,-1)
            res_avg_pr = avg_pr_regret_save
        else:
            res_correct_arm = np.concatenate((res_correct_arm,new_correct_arm.reshape(1,-1)),axis=0)
            res_avg_pr = np.concatenate((res_avg_pr, avg_pr_regret_save), axis=0)
        print(k, new_correct_arm[-1], avg_pr_regret_save[-1,-1])
    np.savez(save_data_path, c_arm=res_correct_arm, avg_pr=res_avg_pr)
    # np.savez(save_data_path, rp=reward_perf,cp=cost_perf, rs=reward_safe,
    #         cs=cost_safe, rb=reward_blend, cb=cost_blend, c_arm=correct_arm_save,
    #         pr=pr_regret_save, avg_pr=avg_pr_regret_save)

def one_trace_blending(env, feat_objective, lam=1.0, delta=0.05, sigma=1.0,
                       theta_max= 10, feat_max=1.0, num_env_interact=1e6,
                       steps_per_epoch=30000, max_ep_len=1000, save_file='',
                       idProcess=0):
    '''
        Blend multiple controllers that implement multiple objectives
    '''
    # print ('starting '+ str(idProcess))
    # Get the dimension of the problem
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    # print(o)
    epochs = int(num_env_interact/steps_per_epoch)
    nArm = len(feat_objective)
    d = feat_objective[0][1](o).shape[0]
    nObj = d
    cum_cost = 0
    # print ('starting '+ str(idProcess))

    # Theorem 2 works for lam >= max {1, feat_max**2}
    assert(lam >= max(1,feat_max**2)), \
                'labmda should be greater than max(1,L^2)'
    def radEllipsis(t):
        return sigma * np.sqrt(d * np.log((1+t*(feat_max**2/lam))/delta)) +\
                np.sqrt(lam) * theta_max

    Vt = lam * np.identity(d)
    Xpast = np.zeros((d,nObj))
    theta_t = np.zeros((d,nObj))
    mu_t = np.zeros((nObj,nArm))
    Ot = np.array([i for i in range(nArm)])
    eps_hat = np.zeros(nArm)
    # eps_true = np.zeros(nArm)

    # Data to be saved
    arm_picked = np.zeros(num_env_interact, dtype=np.uint8)
    pareto_x = np.zeros((nArm,num_env_interact))
    ep_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_rate_save = np.zeros(epochs)

    c_epoch = 0
    c_step = 0
    for t in range(num_env_interact):
        # print(t)
        # select a policy at random from the Pareto set
        xt = np.random.choice(Ot)
        (o2, r, done, info), yt, Xt = feat_map(o, xt, feat_objective, env)
        c = info.get('cost', 0)
        # Temp variable saving X1:t * Y1:t
        for i in range(nObj):
            Xpast[:,i] += Xt[:,xt]*yt[i,xt]
        # Compute Vt+1
        XtXtT = np.matmul(Xt[:,xt:(xt+1)], Xt[:,xt:(xt+1)].T)
        Vt += XtXtT
        # Diagonalize Vt since it's going to be used twice for theta_t+1 and mu
        eVal, P = np.linalg.eig(Vt)
        VtInv = np.matmul(P, np.matmul(np.diag(1.0/eVal), P.T))
        VtSqrtInv = np.matmul(P, np.matmul(np.diag(1.0/np.sqrt(eVal)),P.T))
        # Compute theta_t+1
        for i in range(nObj):
            theta_t[:,i] = np.matmul(VtInv, Xpast[:,i:(i+1)]).flatten()
        # print(theta_t)
        # Compute the radius of the ellipsis
        rE =  radEllipsis(t)
        # Compute the UCB index for each controllers
        for x in range(nArm):
            featX = Xt[:,x:(x+1)]
            n1 = np.linalg.norm(np.matmul(VtSqrtInv,featX))
            tempC = rE * (1.0/n1) * np.matmul(VtInv, featX)
            for i in range(nObj):
                mu_t[i,x] = np.dot(theta_t[:,i] + tempC[:,0], featX[:,0])
        # Compute the pareto gaps
        eps_hat = compute_pgap(mu_t)
        # eps_true = compute_pgap(yt)
        Ot = np.argwhere(eps_hat == np.min(eps_hat)).flatten()

        # Save the useful quantities
        arm_picked[t] = xt
        for xArm in range(nArm):
            pareto_x[xArm, t] = compute_pareto_nd(xArm, yt)

        # Logger informations
        o = o2
        cum_cost += c # Track cumulative cost over training
        ep_ret += r
        ep_cost += c
        ep_len += 1
        terminal = done or (ep_len == max_ep_len)
        if terminal:
            # print(ep_ret, ep_cost)
            if c_epoch < epochs:
                ep_ret_save[c_epoch, c_step] = ep_ret
                cost_ret_save[c_epoch, c_step] = ep_cost
            c_step += 1
            o, r, done, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0

        if t==0 or t % steps_per_epoch !=0:
            continue

        c_epoch += 1
        c_step = 0
        cost_rate_save[c_epoch-1] = cum_cost / (c_epoch*steps_per_epoch)
        print ('-------------------\n'+ str(c_epoch) + ', '+ \
                str(cost_rate_save[c_epoch-1])+ '\n--------------------')
        # if c_epoch == epochs:
        #     break
    np.savez(save_file+str(idProcess), ap=arm_picked, px=pareto_x,
            ret=ep_ret_save, cost=cost_ret_save, crate=cost_rate_save)
    # dictResult[idProcess] = (arm_picked, pareto_x, ep_ret_save, cost_ret_save, \
    #                         cost_rate_save)
    # return arm_picked, pareto_x, ep_ret_save, cost_ret_save, cost_rate_save


if __name__ == '__main__':
    import multiprocessing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath_controllers', '-c', nargs='+', required=True)
    # parser.add_argument('--controllers_name', '-n', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=201)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--exp_name', type=str, default='blending_policy')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--data_contr', type=str, default='')
    parser.add_argument('--idProcess', type=int, default='0')
    args = parser.parse_args()
    # Set the seed for reproducibility
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    # Current environment based on the policies that are extracted
    env,sess = None, None
    # Get the policies and feature estimation
    controller_list = []
    for contr_path in args.fpath_controllers:
        env, get_action, get_objective, sess = load_policy(contr_path,
                    args.itr if args.itr >= 0 else 'last', args.deterministic)
        controller_list.append((get_action, get_objective))

    # outerLoop = 1
    # innerLoop = 1
    # for i in range(outerLoop):
    #     job =[]
    #     # env_list = []
    #     for j in range(innerLoop):
    #         env, _, _, sess = load_policy(args.fpath_controllers[0],
    #                 args.itr if args.itr >= 0 else 'last', args.deterministic)
    #         env.seed(args.seed + (i+1)*(j+1))
    #         p = multiprocessing.Process(target=one_trace_blending,
    #             args=(env, controller_list, 1.0, 0.05, 0.01, 1.5, 1.0, int(100000),
    #                    30000, 1000, args.data_contr,(i+1)*(j+1),))
    #         # env_list.append((env,sess))
    #         job.append(p)
    #         p.start()
    #     for p in job:
    #         p.join()
    # obj1 = np.array([[1.5,1],[0,1]])
    # obj2 = np.array([[1.5,0],[0,1]])
    # obj3 = np.array([[1,0],[0,1]])
    # print(compute_pareto_nd(0, obj1))
    # print(compute_pareto_nd(1, obj1))
    # print(compute_pareto_nd(0, obj2))
    # print(compute_pareto_nd(1, obj2))
    # print(compute_pareto_nd(0, obj3))
    # print(compute_pareto_nd(1, obj3))
    # Start the blending algorithm
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    # blending_algorithm(env, controller_list, lam=1.0, delta=0.2, sigma=0.01,
    #                    theta_max= 1.5, feat_max=1, num_env_interact=int(4e6),
    #                    steps_per_epoch=30000, seed=args.seed, max_ep_len=1000,
    #                    save_freq=50, logger=None, logger_kwargs=logger_kwargs,
    #                    render=args.render, test=args.test, data_path=args.data_contr)
    # blending_algorithm(env, controller_list, lam=1.0, delta=0.2, sigma=0.1,
    #                    theta_max= 1.5, feat_max=1, num_env_interact=int(1e4),
    #                    steps_per_epoch=1000, seed=args.seed, max_ep_len=1000,
    #                    save_freq=50, logger=None, logger_kwargs=logger_kwargs,
    #                    render=args.render, test=args.test, data_path=args.data_contr)
    # blending_alg_np(env, controller_list, lam=1.0, delta=0.2, sigma=0.01,
    #                    theta_max= 1.5, feat_max=1.0, num_run=10,
    #                    max_ep_len=1000, max_env_interact_per_run=1000,
    #                    num_outer_run=10, save_data_path=args.data_contr)
    print (args.seed, args.idProcess)
    env.seed(args.seed + args.idProcess)
    one_trace_blending(env, controller_list, lam=1.0, delta=0.05, sigma=0.01,
                       theta_max= 1.5, feat_max=1, num_env_interact=int(2e6),
                       steps_per_epoch=30000, max_ep_len=1000,
                       save_file= args.data_contr, idProcess=args.idProcess)
