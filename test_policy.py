#!/usr/bin/env python
import tensorflow as tf
import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.run_utils import setup_logger_kwargs

def run_policy(env, get_action, render=False, 
                num_env_interact=int(100000),
                steps_per_epoch=30000, 
                max_ep_len=1000, 
                save_file= '', idProcess=0):
    """Run a pre-recorded policy and save the performance
    achived by that policy.
    """

    # Reset the environment 
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0

    # Number of epochs required
    epochs = int(num_env_interact/steps_per_epoch)
    cum_cost = 0

    # Data to be saved
    ep_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_rate_save = np.zeros(epochs)

    c_epoch = 0
    c_step = 0
    start_time = time.time()
    for t in range(num_env_interact):
        # Render if required
        if render:
            env.render()
            time.sleep(1e-3)

        # Get action associated with the state and step forward in simulation
        a = get_action(o)
        o, r, d, info = env.step(a)

        # Get the resulting cost
        c = info.get('cost', 0)

        # COmpute the episode reward and episode cost
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1
        cum_cost += info.get('cost', 0)

        # Check if we need to reset the environment
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
            np.savez(save_file+str(idProcess), ret=ep_ret_save[:c_epoch,:], 
                cost=cost_ret_save[:c_epoch,:], crate=cost_rate_save[:c_epoch])

        # Do some printing
        print ('IdProcess , Epoch : ', idProcess, c_epoch)
        print ('Time per epoch (seconds) : ', (time.time()-start_time)/(c_epoch+1))
        print(' Cost : ', np.mean(cost_ret_save[c_epoch,:]),
                np.std(cost_ret_save[c_epoch,:]), np.max(cost_ret_save[c_epoch,:]),
                np.min(cost_ret_save[c_epoch,:]))
        print(' Rew : ', np.mean(ep_ret_save[c_epoch,:]),
                np.std(ep_ret_save[c_epoch,:]), np.max(ep_ret_save[c_epoch,:]),
                np.min(ep_ret_save[c_epoch,:]))
        print('----------------------------------------------')

        # Increment epoch and reset the current step
        c_epoch += 1
        c_step = 0

    # Save the last result in a the file
    np.savez(save_file+str(idProcess), ret=ep_ret_save,
                cost=cost_ret_save, crate=cost_rate_save)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--seed', type=int, default=301)
    parser.add_argument('--save_file', type=str, default='')
    parser.add_argument('--idProcess', type=int, default='0')
    parser.add_argument('--num_env_interact', type=int, default=3300000)
    parser.add_argument('--steps_per_epoch', type=int, default=30000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    args = parser.parse_args()

    # Set the seed for reproducibility
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Get the policies
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)

    # Set the seed of the environment 
    print (args.seed, args.idProcess)
    env.seed(args.seed + args.idProcess)

    # Test the given policy
    run_policy(env, get_action, render=args.render,
                num_env_interact=args.num_env_interact, 
                steps_per_epoch=args.steps_per_epoch,
                max_ep_len=args.max_ep_len, save_file= args.save_file,
                idProcess=args.idProcess)

# python test_policy.py safety-starter-agents/data/2020-04-18_ppo_complete/2020-04-18_23-29-38-ppo_complete_s10/ --seed 200 --data_contr ppo_complete_policy --idProcess 0 --norender
# python plot_result.py --logdir ppo_ppoL_blending --legend Blended --colors red --num_traces 10 --ind_traces 0 --steps_per_epoch 30000 --window 5
# ./mprocess_moglb.sh --fpath_controllers safety-starter-agents/data/2020-04-18_ppo_complete/2020-04-18_23-29-38-ppo_complete_s10/ safety-starter-agents/data/2020-04-18_ppo_lagragian_complete/2020-04-18_12-49-35-ppo_lagragian_complete_s10/ --data_contr ppo_ppoL_blending
