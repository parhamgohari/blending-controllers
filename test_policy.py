#!/usr/bin/env python
import tensorflow as tf
import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.run_utils import setup_logger_kwargs

def run_policy(env, get_action, render=False, num_env_interact=int(100000),
                steps_per_epoch=30000, max_ep_len=1000, save_file= '', idProcess=0):
    o, r, done, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    epochs = int(num_env_interact/steps_per_epoch)
    cum_cost = 0
    # Data to be saved
    ep_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_ret_save = np.zeros((epochs, int(steps_per_epoch/max_ep_len)))
    cost_rate_save = np.zeros(epochs)

    c_epoch = 0
    c_step = 0
    for t in range(num_env_interact):
        if render:
            env.render()
            time.sleep(1e-3)
        a = get_action(o)
        o, r, d, info = env.step(a)
        c = info.get('cost', 0)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1
        cum_cost += info.get('cost', 0)
        terminal = done or (ep_len == max_ep_len)
        if terminal:
            if c_epoch < epochs:
                ep_ret_save[c_epoch, c_step] = ep_ret
                cost_ret_save[c_epoch, c_step] = ep_cost
            c_step += 1
            o, r, done, c, ep_ret, ep_len, ep_cost = env.reset(), 0, False, 0, 0, 0, 0
        if t==0 or t % steps_per_epoch !=0:
            continue
        if c_epoch != 0 and c_epoch % 10 == 0:
            np.savez(save_file+str(idProcess), ret=ep_ret_save[:c_epoch,:],
                cost=cost_ret_save[:c_epoch,:], crate=cost_rate_save[:c_epoch])
        c_epoch += 1
        c_step = 0
        cost_rate_save[c_epoch-1] = cum_cost / (c_epoch*steps_per_epoch)
        print ('-------------------\n'+ str(c_epoch) + ', '+ \
                str(cost_rate_save[c_epoch-1])+ '\n--------------------')
    np.savez(save_file+str(idProcess), ret=ep_ret_save,
                cost=cost_ret_save, crate=cost_rate_save)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    # parser.add_argument('--len', '-l', type=int, default=1000)
    # parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--seed', type=int, default=301)
    parser.add_argument('--data_contr', type=str, default='')
    parser.add_argument('--idProcess', type=int, default='0')
    args = parser.parse_args()
    # Set the seed for reproducibility
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    env.seed(args.seed + args.idProcess)
    run_policy(env, get_action, render=not(args.norender),
                num_env_interact=int(4000000), steps_per_epoch=30000,
                max_ep_len=1000, save_file= args.data_contr,
                idProcess=args.idProcess)

# python test_policy.py safety-starter-agents/data/2020-04-18_ppo_complete/2020-04-18_23-29-38-ppo_complete_s10/ --seed 200 --data_contr ppo_complete_policy --idProcess 0 --norender
# python plot_result.py --logdir ppo_ppoL_blending --legend Blended --colors red --num_traces 10 --ind_traces 0 --steps_per_epoch 30000 --window 5
# ./mprocess_moglb.sh --fpath_controllers safety-starter-agents/data/2020-04-18_ppo_complete/2020-04-18_23-29-38-ppo_complete_s10/ safety-starter-agents/data/2020-04-18_ppo_lagragian_complete/2020-04-18_12-49-35-ppo_lagragian_complete_s10/ --data_contr ppo_ppoL_blending
