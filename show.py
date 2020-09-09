# Copyright (c) 2020.06. renweiya. email: weiyren.phd@gmail.com. All rights reserved.
# modified by laker, 2020.06


# -*- coding: utf-8 -*-
import os

import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 使用 GPU 0 or -1 #laker denote
import tensorflow as tf
import numpy as np
import algorithm.common.tf_utils as tf_utils
import time
import random
from gym import spaces

import datetime
DATETIME = "show_"+datetime.datetime.now().strftime('%Y%m%d%H%M')
# read input cmd from standard input device
flags = tf.app.flags

# Game parameter
# flags.DEFINE_string('env_name', 'predator_prey', 'env used')
flags.DEFINE_string('env_name', 'round_up', 'env used')
flags.DEFINE_bool('render', False, 'whether to render the scenario')
flags.DEFINE_integer('seed', 1, 'random seed')

flags.DEFINE_integer('max_step_before_punishment', 8, 'max_step_before_punishment')
flags.DEFINE_bool('reload_prey', True, 'whether to reload the pre-trained prey model')
flags.DEFINE_bool('train_prey', False, 'whether to reload the pre-trained prey model')
flags.DEFINE_string('prey_model_path', './tools/model-prey-23',
                    'path of the pre-trained prey model')
flags.DEFINE_bool('reload_predator', True, 'whether to reload the pre-trained predator model')
flags.DEFINE_string('predator_model_path', './exp_result/pre_trained_prey_20200909100720/saved_models/model-7000',
                    'path of the pre-trained predator model')

# Training parameters
flags.DEFINE_bool('learning', False, 'train the agents')
flags.DEFINE_string('exp_name', 'rounding', 'exp_name')
flags.DEFINE_string('predator_policy', 'pgddpg', 'predator_policy: [pgddpg, ddpg]') 
flags.DEFINE_string('prey_policy', 'ddpg', 'prey_policy: [random, fixed, ddpg]')
flags.DEFINE_integer('episodes', 500000, 'maximum training episode')#500000

flags.DEFINE_float('ddpg_plr', 0.01, 'policy learning rate')
flags.DEFINE_float('ddpg_qlr', 0.001, 'critic learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float('tau', 0.01, 'target network update frequency')
flags.DEFINE_integer('target_update_interval', 1, 'target network update frequency')
flags.DEFINE_float('return_confidence_factor', 0.7, 'return_confidence_factor')

flags.DEFINE_integer('n_train_repeat', 1, 'repeated sample times at each training time')
flags.DEFINE_integer('save_checkpoint_every_epoch', 5000, 'save_checkpoint_every_epoch')#5000
flags.DEFINE_integer('plot_reward_recent_mean', 1000, 'show the avg reward of recent 200 episode')
flags.DEFINE_bool('save_return', True, 'save trajectory Return by default')
flags.DEFINE_float('lambda1', 0., 'n-step return')
flags.DEFINE_float('lambda1_max', 1., 'n-step return')
flags.DEFINE_float('lambda2', 1e-6, 'coefficient of regularization')
#explore settings
flags.DEFINE_float('n_steps_annealing', 1.0, 'n_steps_annealing')#laker 1.0
flags.DEFINE_float('delta_n_steps_annealing', 0.0, 'delta_n_steps_annealing')#laker 0.0

#
flags.DEFINE_integer('num_adversaries', 3, 'num_adversaries')#3
flags.DEFINE_integer('num_good_agents', 1, 'num_good_agents')#1

flags.DEFINE_float('multi_critic_beta', 0.9, 'multi_critic_beta')

# #
flags.DEFINE_integer('max_episode_len', 200, 'maximum step of each episode')#200
flags.DEFINE_integer('train_every', 200, 'train every x sim steps')#train_every
flags.DEFINE_integer('batch_size', 1024, 'batch size')#1024
# experience replay
flags.DEFINE_integer('buffer_size', 300000, 'buffer size')#laker 300000
flags.DEFINE_integer('min_buffer_size', 30000, 'minimum buffer size before training')#laker 30000


#debug
# flags.DEFINE_integer('max_episode_len', 3, 'maximum step of each episode')#200
# flags.DEFINE_integer('train_every', 3, 'train every x sim steps')#train_every
# flags.DEFINE_integer('batch_size', 1024, 'batch size')#1024
# flags.DEFINE_integer('buffer_size', 3000, 'buffer size')#laker 300000
# flags.DEFINE_integer('min_buffer_size', 3, 'minimum buffer size before training')#laker 30000


# prioritized
flags.DEFINE_bool('prioritized_er', False, 'whether to use prioritized ER')
flags.DEFINE_float('alpha', 0.6, 'how much prioritization is used (0 - no prioritization, 1 - full prioritization)')
flags.DEFINE_float('beta', 0.4, 'To what degree to use importance weights (0 - no corrections, 1 - full correction)')

# Net structure
flags.DEFINE_integer('num_units', 128, 'layer neuron number')#laker 128
flags.DEFINE_integer('h_layer_num', 3, 'hidden layer num')

# Model saving dir

FLAGS = flags.FLAGS  # alias


def make_env(scenario_name, max_step_before_punishment):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from env.multiagent.environment import MultiAgentEnv
    import env.multiagent.scenarios as scenarios

    # load scenario from script
    # scenario = scenarios.load(scenario_name + ".py").Scenario() # modify
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    scenario.max_step_before_punishment = max_step_before_punishment
    
    #add by laker
    scenario.num_adversaries=FLAGS.num_adversaries
    scenario.num_good_agents=FLAGS.num_good_agents
    scenario.successed_round_up=FLAGS.num_adversaries
    scenario.good_colors = [np.array([0x00, 0x99, 0xff]) / 255] * FLAGS.num_good_agents
    #add end
    print('==============================================================')
    print('max_step_before_punishment: ', scenario.max_step_before_punishment)
    print('==============================================================')

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        info_callback=scenario.collision_number,
                        done_callback=scenario.done,
                        shared_viewer=True,#modify share view or not
                        other_callbacks=[scenario.set_arrested_pressed_watched])
    return env


def build_agents(action_dim_n, observation_dim_n, policies_name):
    '''
    build agents
    :param action_dim_n:
    :param observation_dim_n:
    :param policies_name:
    :return:
    '''
    from algorithm.trainer import SimpleAgentFactory
    agents = []
    # obs_shape_n = [[dim] for dim in observation_dim_n]
    
    for agent_idx, policy_name in enumerate(policies_name):
        agents.append(SimpleAgentFactory.createAgent(agent_idx, policy_name, observation_dim_n[agent_idx], action_dim_n[agent_idx], FLAGS))
    return agents


def reload_previous_models(session, env):
    import gc
    # preload prey policy
    if FLAGS.reload_prey:
        prey_vars = []
        for idx in range(FLAGS.num_adversaries, env.n):#range(2,5)
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_{}'.format(idx))
            prey_vars += var

        saver_prey = tf.train.Saver(var_list=prey_vars)
        saver_prey.restore(session, FLAGS.prey_model_path)

        print('[prey] successfully reload previously saved ddpg model({})...'.format(FLAGS.prey_model_path))
        del saver_prey
        gc.collect()
        # all the predator using the same policy
        # best_agent = agents[base_kwargs['num_adversaries']]
        # for i in range(base_kwargs['num_adversaries'], env.n):
        #     agents[i] = best_agent

    if FLAGS.reload_predator:
        predator_vars = []
        for idx in range(FLAGS.num_adversaries):
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_{}'.format(idx))
            predator_vars += var
        saver_predator = tf.train.Saver(var_list=predator_vars)
        saver_predator.restore(session, FLAGS.predator_model_path)
        print('[predator] successfully reload previously saved RL model({})...'.format(
            FLAGS.predator_model_path
        ))
        del saver_predator
        gc.collect()


def train():
    # init env
    env = make_env(FLAGS.env_name, FLAGS.max_step_before_punishment)
    env = env.unwrapped
    # set env seed
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    print("Using seed {} ...".format(FLAGS.seed))

    print('There are total {} agents.'.format(env.n))
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [2] * env.n
    print('obs_shape_n: ', obs_shape_n)  # [16, 16, 16, 14]
    print(action_shape_n)  # [5, 5, 5, 5]

    adv_policies = [FLAGS.predator_policy] * FLAGS.num_adversaries
    good_policies = [FLAGS.prey_policy] * FLAGS.num_good_agents
    print(adv_policies + good_policies)

    with tf_utils.make_session().as_default() as sess:
        # init agents
        agents = build_agents(action_shape_n, obs_shape_n, adv_policies + good_policies)

        # init tf summaries
        # reload previous prey and predator model
        reload_previous_models(session=sess, env=env)

        # Initialize uninitialized variables.
        tf_utils.initialize(sess=sess)
        # assert using same session
        same_session(sess, agents)
        #  make the tensor graph unchangeable
        sess.graph.finalize()

        # collect some statistical data
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_episode_rewards = [[0.0] for _ in range(env.n)]
        
        total_times = 0

        obs_n = env.reset() #observations 18,18,16,16,16
        episode_step = 0  # step for each episode
        train_step = 0  # total training step
        t_start = time.time()
        print('Starting iterations...')
        sucess_rate=[]
        sucess_record=[]
        explore_pobility=1.0 #float
        gama_gussian_=0.0
        start_decay=False
        start_decay_n=0
        episode_step_all=[]
        while len(episode_rewards) <= FLAGS.episodes:#140000
            #explore pobility

            # increment global step counter
            train_step += 1
            # explore settings-laker
            temp_random_p1=np.random.rand()
            if temp_random_p1<explore_pobility:# explore more at first
                step_explore_=1.0
            else: #no  explore
                step_explore_=0.0

            temp_random_p2=np.random.rand()
            if temp_random_p2>0.5: # main direction of the explore
                explore_direction_= 1.0
            else:# explore
                explore_direction_=-1.0

            #get actions
            if 1:#FLAGS.train_prey:
                action_2_dim_n = [agent.get_actions(observations=[obs], single=True, step_explore=0.0, explore_direction=explore_direction_,gama_gussian=gama_gussian_) for agent, obs in zip(agents, obs_n)]
            else:
                #no noise for prey
                action_2_dim_n=[]
                count_temp=0
                for agent, obs in zip(agents, obs_n):
                    if count_temp==3:
                        # action_prey=agent.get_actions(observations=[obs], single=True, step_explore=step_explore_, explore_direction=explore_direction_,gama_gussian=gama_gussian_)

                        # action_prey=np.maximum(action_prey,-0.5)
                        # action_prey=np.minimum(action_prey,0.5)

                        # action_2_dim_n.append(action_prey)
                        action_2_dim_n.append(agent.get_actions(observations=[obs], single=True, step_explore=0.0, explore_direction=explore_direction_,gama_gussian=gama_gussian_))    
                    else:    
                        action_2_dim_n.append(agent.get_actions(observations=[obs], single=True, step_explore=step_explore_, explore_direction=explore_direction_,gama_gussian=gama_gussian_))    
                    count_temp=count_temp+1
            
            action_n = [[0, a[0], 0, a[1], 0] for a in action_2_dim_n]

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n, restrict_move=True)

            info_n = info_n['n']
            episode_step += 1

            done = all(done_n)  # 达到任务
            terminal = (episode_step >= FLAGS.max_episode_len)  # 最大步数
            ended = done or terminal

            # modify: add by laker: render after the step. important
            show_fre=100
            show_fre_render=1
            if FLAGS.render and len(episode_rewards) % show_fre_render == 0: # modify
                time.sleep(0.3)
                env.render(mode='no-human')

            #add by laker
            if ended:
                env._reset_render()#reset render-laker
                sucess_rate.append(done)
                if start_decay and explore_pobility>0.05:
                    start_decay_n=start_decay_n+1
                    explore_pobility=1.0 #pow(0.999,0.01*start_decay_n)
                if start_decay and explore_pobility<0.05:
                    explore_pobility=0.05
                
                if gama_gussian_<0.05:
                    gama_gussian_=0.0
                else:
                    gama_gussian_=pow(0.9,0.01*start_decay_n)  
                

            if len(episode_rewards) % show_fre_render == 0:
                if ended==True:
                    print("cur_episode:",len(episode_rewards)," ,explore_pobility:",explore_pobility," ,gama_gussian_:",gama_gussian_)
                    print("done?",done,terminal)
                    # print("episode_step:",episode_step)
                    episode_step_all.append(episode_step)
                    print("mean episode steps:",sum(episode_step_all)/len(episode_step_all))
                    length_k=show_fre
                    if len(sucess_rate)>length_k:
                        print("mean sucess rate",sum(sucess_rate)/len(sucess_rate))
                        
                        #print("sucess rate",sum(sucess_rate[-length_k:])/len(sucess_rate[-length_k:]))
                        sucess_record.append(sum(sucess_rate[-length_k:])/len(sucess_rate[-length_k:]))
                
                if episode_step >= FLAGS.max_episode_len:
                    print("max steps",episode_step)

            # collect experience
            if FLAGS.learning:
                for i, agent in enumerate(agents):

                    #modify by laker: all learning
                    if FLAGS.train_prey==True:
                        agent.experience(obs_n[i], action_2_dim_n[i], rew_n[i], new_obs_n[i], ended)                
                    # prey is fixed---#modify by laker: all learning
                    elif i < FLAGS.num_adversaries:
                        agent.experience(obs_n[i], action_2_dim_n[i], rew_n[i], new_obs_n[i], ended)

            # step forward observations
            obs_n = new_obs_n

            # TODO: 这里记录每一轮最大reward
            # record some analysis information
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] =round(episode_rewards[-1]+ rew,10) #lastest reward sum for each step of all agent
                agent_episode_rewards[i].append(rew)
                agent_rewards[i][-1] =round(agent_rewards[i][-1]+ rew,10) #lastest reward sum for each step of each agent


            if ended: #reset
                # print log for debugging......
                if len(episode_rewards) % show_fre == 0:
                    print('process {}, episode {}: '.format(os.getpid(), len(episode_rewards)))
                    print("")
                    print("")
                # reset environment
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)  # reset sum rewards #laker another episode
                for idx, a in enumerate(agent_rewards):  # reset each agent's reward
                    a.append(0)
                agent_episode_rewards = [[0.0] for _ in range(env.n)]

        print("finished")
        # close sess
        sess.close()

# for debug below ..........................................................
def same_session(sess, agents):
    for agent in agents[:FLAGS.num_adversaries]:
        if sess != agent.get_session():
            print("Session error (diff tf session)")
    print("The same session.........................")


if __name__ == '__main__':
    train()
