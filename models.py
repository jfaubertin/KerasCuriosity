import numpy as np

#=========================================================================#

from keras.models import Model, Sequential
from keras.layers import Input, Concatenate, Convolution2D, Flatten, Dense, Reshape #, Activation
from keras import backend as K

#=========================================================================#

# simple action model (far from optimal)
def build_actor_model(state_shape, nb_actions):
    model = Sequential()

    model.add(Reshape(state_shape[1::], input_shape=state_shape))
#    model.add(Reshape([224,256,3], input_shape=state_shape)) # FIXME: temporary patch; shouldn't be so environment dependent

#    model.add(Convolution2D(8, (1, 1), strides=(1, 1), name='conv.1x1', padding='same', activation='relu' ))
    model.add(Convolution2D(8, (4, 4), strides=(2, 2), name='conv1', padding='same', activation='relu' ))
    model.add(Convolution2D(8, (4, 4), strides=(2, 2), name='conv2', padding='same', activation='relu' ))
    model.add(Convolution2D(8, (4, 4), strides=(2, 2), name='conv3', padding='same', activation='relu' ))
    model.add(Convolution2D(8, (4, 4), strides=(2, 2), name='conv4', padding='same', activation='relu' ))

    model.add(Flatten())

    # fc1 is intentionally smaller to reduce parameters
    # feel free to increase this if you have the hardware
    model.add(Dense( 16, name='fc1', activation='relu'))
    model.add(Dense(128, name='fc2', activation='relu'))
    model.add(Dense(nb_actions, name='output', activation='softmax'))

    # print(model.summary())

    return model

#=========================================================================#

# state_shape=[224,256,3]
def build_fmap(state_shape, name_prefix='fmap.'): # , output_shape, nb_actions
    print("models.build_fmap()")

    #state_shape=[224,256,3]
    #print "state_shape:"
    #print state_shape

    # (input) 224x256(x3) / 16x16(via 5 conv layers) = 14x16(x8) = 1792 (output)

    # fmap = (input) 224x256(x3) / 16x16(via 5 conv layers) = 14x16(x8) (output)


    inputs = Input(shape=state_shape)
    x = inputs
#    x = Reshape([224,256,3])(x) # FIXME: temporary patch; shouldn't be so environment dependent

    # optional - uncomment to scan for 16 colors first
#    x = Convolution2D(16, (1, 1), strides=(1, 1), name='conv.1x1', padding='same', activation='relu')(x)
#    x = Convolution2D(8, (4, 4), strides=(2, 2), name=name_prefix+'conv1', padding='same', activation='relu')(x)
    x = Convolution2D(8, (4, 4), strides=(2, 2), name=name_prefix+'conv1', padding='same', activation='relu' )(x)
    x = Convolution2D(8, (4, 4), strides=(2, 2), name=name_prefix+'conv2', padding='same', activation='relu' )(x)
    x = Convolution2D(8, (4, 4), strides=(2, 2), name=name_prefix+'conv3', padding='same', activation='relu' )(x)
    x = Convolution2D(8, (4, 4), strides=(2, 2), name=name_prefix+'conv4', padding='same', activation='relu' )(x)
#    x = Convolution2D(8, (4, 4), strides=(2, 2), name=name_prefix+'conv5', activation='relu' )(x)

    # Flatten so models that include this one don't have to
    x = Flatten(name=name_prefix+'flat')(x)

    model = Model(inputs, x, name=name_prefix+'feature_map')

    return model

#=========================================================================#

# Intrinsic Curiosity Model
# Inverse model: predicts action given past and current state
def build_inverse_model(fmap1, fmap2, num_actions):
    print("models.build_inverse_model()")

    #======================#
    # input = prev state + current state
    # concat (prev state + current state)
    # output = action taken between states
    #======================#

    # prepare inputs
    obs1=fmap1
    obs2=fmap2
    x = Concatenate()([obs1.output, obs2.output])

    #======================#

    # fc1 is intentionally smaller to reduce parameters
    # feel free to increase this if you have better hardware
    x = Dense(16, name='icm_i.fc1', activation='relu')(x)
    x = Dense(128, name='icm_i.fc2', activation='relu')(x)

    #======================#

    x = Dense(num_actions, name='icm_i.output', activation='sigmoid')(x)

    i_model = Model([obs1.input,obs2.input], x, name='icm_inverse_model')

    #print(i_model.summary())
    return i_model

#=========================================================================#

# Intrinsic Curiosity Model
# Forward model: predicts future state given current state and action
def build_forward_model(fmap, num_actions):
    print("models.build_forward_model()")

    #======================#
    # input = current state + action
    # concat (flattened state + action)
    # output = next state
    #======================#

    # prepare inputs
    obs1=fmap
    act1=Input(shape=(num_actions,))
    x = Concatenate()([obs1.output, act1])

    #======================#

    # fc1 and fc3 are intentionally smaller to reduce parameters
    # feel free to increase this if you have better hardware
    x = Dense( 32, name='icm_f.fc1', activation='relu')(x)
    x = Dense(128, name='icm_f.fc2', activation='relu')(x)
    x = Dense( 32, name='icm_f.fc3', activation='relu')(x)

    #======================#

    output_shape = obs1.output_shape[1]
    x = Dense( output_shape, name='icm_f.output', activation='linear')(x)

    f_model = Model([obs1.input,act1], x, name='icm_forward_model')

    #print(f_model.summary())
    return f_model

#=========================================================================#

if __name__ == "__main__":
    print("models.main()")

    state_shape=(224,256,3)
    nb_actions=24

    # CREATE FEATURE MAP
    fmap = build_fmap(state_shape)
    fmap2 = build_fmap(state_shape, name_prefix='fmap2.')
    print "feature map: "
    print(fmap.summary())
#    exit()

    # CREATE MODELS
    print "CREATE MODELS..."

    inv_model = build_inverse_model(fmap, fmap2, nb_actions)
    print "inv_model: "
    print(inv_model.summary())

    fwd_model = build_forward_model(fmap, nb_actions)
    print "fwd_model: "
    print(fwd_model.summary())

    actor_model = build_actor_model((1,)+state_shape, nb_actions)
    print "actor_model: "
    print(actor_model.summary())
#    exit()

    # TEST MODELS
    print "TEST MODELS..."
    obs1 = np.random.rand( 224,256,3 )
    obs2 = np.random.rand( 224,256,3 )
    icm_action = np.zeros(nb_actions)
    icm_action[1]=1
    print "icm_action: "
    print icm_action
    print icm_action.shape

    print "inv_model prediction: "
    print inv_model.predict([np.expand_dims(obs1,0),np.expand_dims(obs2,0)])       # output = icm_action

    print "fwd_model prediction: "
    print fwd_model.predict([np.expand_dims(obs1,0),np.expand_dims(icm_action,0)]) # output = obs2

    print "act_model prediction: "
    print actor_model.predict([np.expand_dims(np.expand_dims(obs1,0),0)]) # output = action
#    exit()


    print("Done.")
    exit()

#=========================================================================#
