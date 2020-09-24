import numpy as np
import time
import tensorflow as tf
import joblib
import os
import os.path as osp
from safe_rl.utils.logx import restore_tf_graph

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
    # get_objective = lambda x: np.array([[get_v(x)],[get_vc(x)]])
    get_objective = lambda x: np.array([[get_v(x)],[-get_vc(x)]])
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
            # feat_t = featVal
        else:
            featVal = objective_fun(zt)
            # feat_t = np.concatenate((feat_t, featVal), axis=1)
            feat_t = np.concatenate((feat_t, featVal/np.linalg.norm(featVal)),
                                        axis=1)
        list_act.append(curr_act(zt))
    o2_true, r_true, done_true, info_true, yt = env.step_simulate(xt, list_act)
    return (o2_true, r_true, done_true, info_true), yt, feat_t


def compute_loss(objective):
    '''
    Compute the gap given a matrix whose column represent
    each arms and the row are the different objectives
    @Return     :   an array containing the pareto gap for each arm
    '''
    res = np.zeros(objective.shape[1])
    for x in range(res.shape[0]):
        temp_x = objective - np.tile(objective[:,x:(x+1)],2)
        res[x] = np.maximum(np.max(np.max(temp_x, axis=1)),0)
    return res

def compute_pareto_gap(xt, objective):
    '''
    Compute the pareto non-dominace gap. Useful to show the
    subregret bounds
    '''
    other_r = objective[0][1-xt]
    other_c = objective[1][1-xt]
    curr_r = objective[0][xt]
    curr_c = objective[1][xt]

    diff_r = other_r - curr_r
    diff_c = other_c - curr_c
    if diff_r*diff_c >= 0:
        return np.maximum(np.maximum(diff_r,diff_c), 0)
    else:
        return 0

def compute_pareto_nd(xt, objective):
    '''
    Compute the pareto non-dominace gap. Useful to show the
    subregret bounds
    '''
    other_r = objective[0][1-xt]
    other_c = objective[1][1-xt]
    curr_r = objective[0][xt]
    curr_c = objective[1][xt]

    diff_r = other_r - curr_r
    diff_c = other_c - curr_c
    if diff_r*diff_c >= 0:
        return np.maximum(np.maximum(diff_r,diff_c), 0)
    else:
        return 0

def compute_pgap(objective):
    '''
    Compute the gap given a matrix whose column represent
    each arms and the row are the different objectives
    @Return     :   an array containing the pareto gap for each arm
    '''
    res = np.zeros(objective.shape[1])
    for x in range(res.shape[0]):
        temp_x = objective - np.tile(objective[:,x:(x+1)],2)
        res[x] = np.maximum(np.max(np.max(temp_x, axis=1)),0)
    return res


def one_trace_blending(env, feat_objective, lam=1.0, delta=0.05, sigma=1.0,
                       theta_max= 10, feat_max=1.0, num_env_interact=1e6,
                       steps_per_epoch=30000, max_ep_len=1000, save_file='',
                       idProcess=0, render=False):
    '''
        Blend multiple controllers that maximize multiple objectives
    '''
    # print ('starting '+ str(idProcess))
    # Reset the environment and get the dimension of the problem
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    epochs = int(num_env_interact/steps_per_epoch)
    nArm = len(feat_objective)
    d = feat_objective[0][1](o).shape[0]
    nObj = d
    cum_cost = 0

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

    # Data to be saved
    arm_picked = np.zeros(num_env_interact, dtype=np.uint8)
    pareto_x = np.zeros((nArm,num_env_interact))
    ep_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_rate_save = np.zeros(epochs)
    save_theta_rev_cost = np.zeros((num_env_interact, 2,d))
    save_context = np.zeros((num_env_interact, d))
    save_rev_cost_im = np.zeros((num_env_interact, 2))

    c_epoch = 0
    c_step = 0
    start_time = time.time()
    for t in range(num_env_interact):
        # Render if required
        if render:
            env.render()
            time.sleep(1e-3)

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

        # Store the variable for linear approximation proof
        save_theta_rev_cost[t,:,:] = theta_t[:,:]
        save_context[t,:] = Xt[:,xt]
        save_rev_cost_im[t,:] = yt[:, xt]


        # Compute theta_t hat
        for i in range(nObj):
            theta_t[:,i] = np.matmul(VtInv, Xpast[:,i:(i+1)]).flatten()

        # Compute the radius of the ellipsis
        rE =  radEllipsis(t)

        # Compute the UCB index for each controllers
        for x in range(nArm):
            featX = Xt[:,x:(x+1)]
            tempVect = np.matmul(VtInv, featX)
            n1 = np.sqrt(np.dot(featX[:,0], tempVect[:,0]))
            tempC = rE * (1.0/n1) * tempVect
            for i in range(nObj):
                mu_t[i,x] = np.dot(theta_t[:,i] + tempC[:,0], featX[:,0])

        # Compute the pareto gaps
        eps_hat = compute_pgap(mu_t)

        # Compute the new pareto set based on the pareto gaps
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
            if c_epoch < epochs:
                ep_ret_save[c_epoch, c_step] = ep_ret
                cost_ret_save[c_epoch, c_step] = ep_cost
            c_step += 1
            o, r, done, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0

        # Looging only at each epochs
        if t==0 or t % steps_per_epoch !=0:
            continue

        # Compute and save the cost rate
        cost_rate_save[c_epoch] = cum_cost / ((c_epoch+1)*steps_per_epoch)

        # Save the data into a file every 'save_epochs' epochs
        save_epochs = 5
        if c_epoch != 0 and c_epoch % save_epochs == 0:
            np.savez(save_file+str(idProcess), ap=arm_picked[:t],
                px=pareto_x[:,:t], ret=ep_ret_save[:c_epoch,:],
                cost=cost_ret_save[:c_epoch,:],
                crate=cost_rate_save[:c_epoch], context=save_context,
                theta=save_theta_rev_cost, revcost=save_rev_cost_im)

        # Do some printing
        print ('IdProcess , Epoch : ', idProcess, c_epoch)
        print ('Time per epoch (seconds) : ', (time.time()-start_time)/(c_epoch+1))
        print('Cost : ', np.mean(cost_ret_save[c_epoch,:]),
                np.std(cost_ret_save[c_epoch,:]), np.max(cost_ret_save[c_epoch,:]),
                np.min(cost_ret_save[c_epoch,:]))
        print('Rew : ', np.mean(ep_ret_save[c_epoch,:]),
                np.std(ep_ret_save[c_epoch,:]), np.max(ep_ret_save[c_epoch,:]),
                np.min(ep_ret_save[c_epoch,:]))
        print('Normalized theta : ', theta_t[:,0]/np.linalg.norm(theta_t[:,0]),
                theta_t[:,1]/np.linalg.norm(theta_t[:,1]))
        print('----------------------------------------------')
        # Increment epoch and reset the current step
        c_epoch += 1
        c_step = 0

    # Save the last result in a the file
    np.savez(save_file+str(idProcess), ap=arm_picked, px=pareto_x,
            ret=ep_ret_save, cost=cost_ret_save, crate=cost_rate_save,
            context=save_context, theta=save_theta_rev_cost,
            revcost=save_rev_cost_im)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath_controllers', '-c', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=201)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_file', type=str, default='')
    parser.add_argument('--idProcess', type=int, default='0')
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--theta_max', type=float, default=1.5)
    parser.add_argument('--feat_max', type=float, default=1.0)
    parser.add_argument('--num_env_interact', type=int, default=3300000)
    parser.add_argument('--steps_per_epoch', type=int, default=30000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
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

    # Set the seed of the environment
    print (args.seed, args.idProcess)
    env.seed(args.seed + args.idProcess)

    # Execute the algorithm
    one_trace_blending(env, controller_list, lam=args.lam, delta=args.delta,
                        sigma=args.sigma, theta_max= args.theta_max,
                        feat_max=args.feat_max, render=args.render,
                        num_env_interact=args.num_env_interact,
                        steps_per_epoch=args.steps_per_epoch,
                        max_ep_len=args.max_ep_len, save_file= args.save_file,
                        idProcess=args.idProcess)
