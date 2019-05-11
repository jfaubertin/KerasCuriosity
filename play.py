import numpy as np
import gym
import os
#import os.path

from keras.optimizers import Adam
import keras.backend as K

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy

#=========================================================================#

import ppaquette_gym_super_mario

#=========================================================================#

# local
from md2d import MD2D_ActionWrapper
from models import build_actor_model, build_fmap, build_inverse_model, build_forward_model

#=========================================================================#

#env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

ENV_NAME = 'ppaquette/SuperMarioBros-1-2-v0'
inv_weights_fname = '{}_inv_weights.h5f'.format("SMB")
fwd_weights_fname = '{}_fwd_weights.h5f'.format("SMB")
agent_weights_fname = '{}_agent_weights.h5f'.format("SMB")

###########################################################################

NES_buttons = {
     0: [0, 0, 0, 0, 0, 0],  # Do Nothing
     1: [1, 0, 0, 0, 0, 0],  # Up
     2: [0, 1, 0, 0, 0, 0],  # Left
     3: [0, 0, 1, 0, 0, 0],  # Down
     4: [0, 0, 0, 1, 0, 0],  # Right
     5: [0, 0, 0, 0, 1, 0],  # A
     6: [0, 0, 0, 0, 0, 1],  # B
     7: [0, 0, 0, 0, 1, 1],  # A + B
     8: [1, 0, 0, 1, 0, 0],  # Up    + Right
     9: [1, 1, 0, 0, 0, 0],  # Up    + Left
    10: [0, 0, 1, 1, 0, 0],  # Down  + Right
    11: [0, 1, 1, 0, 0, 0],  # Down  + Left
    12: [1, 0, 0, 0, 1, 0],  # Up    + A
    13: [0, 1, 0, 0, 1, 0],  # Left  + A
    14: [0, 0, 1, 0, 1, 0],  # Down  + A
    15: [0, 0, 0, 1, 1, 0],  # Right + A
    16: [1, 0, 0, 0, 0, 1],  # Up    + B
    17: [0, 1, 0, 0, 0, 1],  # Left  + B
    18: [0, 0, 1, 0, 0, 1],  # Down  + B
    19: [0, 0, 0, 1, 0, 1],  # Right + B
    20: [1, 0, 0, 0, 1, 1],  # Up    + A+B
    21: [0, 1, 0, 0, 1, 1],  # Left  + A+B
    22: [0, 0, 1, 0, 1, 1],  # Down  + A+B
    23: [0, 0, 0, 1, 1, 1],  # Right + A+B
  }



###########################################################################


#=========================================================================#

if __name__ == "__main__":
    env = gym.make(ENV_NAME) # Load Game
    env = MD2D_ActionWrapper(env,NES_buttons) # MultiDiscrete to Discrete so SARSAAgent will work
#    env = MarioActionSpaceWrapper(env)
    #env = ProcessFrame84(env)

    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    #nb_actions = 14

#    print "env.observation_space.shape: "
#    print env.observation_space.shape
    observation_shape = env.observation_space.shape
    #observation_shape = (224, 256, 3)

    # repeat action to save processing / run faster
    action_repeat = 2 #6

#=========================================================================#

    fmap  = build_fmap( observation_shape)  # shared feature map - important!
    fmap2 = build_fmap( observation_shape,name_prefix='fmap2.') 

    # predicts action from current state and future state
    inverse_model = build_inverse_model(fmap,fmap2,nb_actions)
    inverse_model.compile(Adam(lr=1e-3), loss='mse', metrics=['mse'])
#    print(inverse_model.summary())

    # predicts future state from current state and action
    forward_model = build_forward_model(fmap,nb_actions)
    forward_model.compile(Adam(lr=1e-3), loss='mse', metrics=['mse'])
#    print(forward_model.summary())

    model = build_actor_model((1,)+observation_shape, nb_actions)
#    print(model.summary())

    policy = BoltzmannQPolicy()
    agent = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=3, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    agent.reset_states()

#=========================================================================#

    # re-use weights if possible
    if (os.path.isfile( inv_weights_fname )):
        inverse_model.load_weights( inv_weights_fname );

    if (os.path.isfile( fwd_weights_fname )):
        forward_model.load_weights( fwd_weights_fname );

    if (os.path.isfile( agent_weights_fname )):
        agent.load_weights( agent_weights_fname );
#    else:
        # FIXME: this bit is necessary or agent does nothing???
        # probably initializes values or something
#    agent.fit(env, nb_steps=20, visualize=False) 
    agent.training = True # IMPORTANT!!! or it doesn't learn

#=========================================================================#

    episode_count = 1000
    reward = 0
    done = False

    for i in range(episode_count):
        print "episode=%d" % i
        obs_now = env.reset()
        obs_last=obs_now
        obs_now, env_reward, done, _ = env.step(0)
#        print "obs_now = "
#        print obs_now.shape


        while not done:
#            print ""
#            print "step"

            # agent takes action
            action = agent.forward( obs_now )

            # save previous step for use by ICM
            ob_last=obs_now

            # repeat action to save processing / run faster
            for i in range(action_repeat):
                obs_now, env_reward, done, _ = env.step(action)
#            print "env_reward = %f" % env_reward
#            print "obs_now = "
#            print obs_now.shape

            icm_action = np.zeros(nb_actions)
            icm_action[action]=1

            # train/test inverse model
            inv_loss = inverse_model.train_on_batch([ np.expand_dims(obs_last,0), np.expand_dims(obs_now,0) ], [np.expand_dims(icm_action,0)] )
#            print "inv"
#            print inv_loss

            # FIXME: get fwd_model to predict fmap2.output
            # FIXME: somehow get fmap2 output so fwd_model to predict

            # train/test forward model
            features_now = fmap.predict(np.expand_dims(obs_now,0))
            fwd_loss = forward_model.train_on_batch([ np.expand_dims(obs_last,0), np.expand_dims(icm_action,0)], [features_now] )
#            print "fwd"
#            print fwd_loss

            # calculate agent reward based on forward model loss
            r_intr = (fwd_loss[0] ** 0.5) /100
#            print "r_intr = %d" % r_intr

            # TODO: could use a ratio for intrinsic vs environment reward
            reward = r_intr # + env_reward
#            print "reward = %f" % reward

            # apply reward
            action = agent.backward(reward, done)

            if done:
                inverse_model.save_weights( inv_weights_fname, overwrite=True)
                forward_model.save_weights( fwd_weights_fname, overwrite=True)
                agent.save_weights( agent_weights_fname, overwrite=True)
                break
    env.close()

    exit();

#=========================================================================#
# NOTES:
# x start with random actions + test observations
# x train/test inverse model
# x train/test forward model
# x calculate agent reward (based on forward model)
# x change agent to sarsa
# x add forward pass
# x add backward pass
# x save/load weights
# x repeat actions to speed up simulation
#=========================================================================#
