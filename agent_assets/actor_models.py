import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from . import A_hparameters as hp
"""
Actor models takes one input:
    1. encoded states
and returns an action.

Critic model functions should take following arguments:
    1. observation_space
    2. action_space : Box
    3. encoder_f
"""

def actor_simple_dense(observation_space, action_space, encoder_f):
    encoded_state, encoder_inputs = encoder_f(observation_space)
    s = layers.Flatten(name='actor_flatten_state')(encoded_state)
    action_shape = action_space.shape
    action_num = tf.math.reduce_prod(action_shape)
    action_range = action_space.high - action_space.low
    action_middle = (action_space.low + action_space.high)/2

    x = layers.Dense(256, activation='relu',
                     name='actor_dense1')(s)
    x = layers.Dense(128, activation='relu',
                     name='actor_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='actor_dense3')(x)
    x = layers.Dense(action_num, activation=hp.Actor_activation,
                     name='actor_dense4',)(x)
    x = layers.Reshape(action_space.shape, name='actor_reshape')(x)
    outputs = x*action_range/2 + action_middle
    outputs = layers.Activation('linear',dtype='float32',
                                         name='actor_float32')(outputs)

    model = keras.Model(
        inputs=encoder_inputs,
        outputs=outputs,
        name='actor'
    )

    return model

def actor_vmpo_dense(observation_space, action_space, encoder_f):
    encoded_state, encoder_inputs = encoder_f(observation_space)
    s = layers.Flatten(name='actor_flatten_state')(encoded_state)
    action_shape = action_space.shape
    action_num = tf.math.reduce_prod(action_shape)
    action_range = action_space.high - action_space.low
    action_middle = (action_space.low + action_space.high)/2

    x = layers.Dense(256, activation='relu',
                     name='actor_dense1')(s)
    x = layers.Dense(128, activation='relu',
                     name='actor_dense2')(x)
    x = layers.Dense(64, activation='relu',
                     name='actor_dense3')(x)

    mu = layers.Dense(action_num, activation=hp.Actor_activation,
                     name='actor_mu_dense',)(x)
    mu = layers.Reshape(action_space.shape, name='actor_reshape')(mu)
    mu = mu*action_range/2 + action_middle
    mu = layers.Activation('linear',dtype='float32',
                                         name='actor_mu_float32')(mu)

    chol_flat = layers.Dense((action_num*(action_num-1))//2,
                            activation='linear', name='actor_chol_flat')(x)
    chol_tri = tfp.math.fill_triangular(chol_flat, name='actor_chol_tri')

    diag_i = tf.tile(tf.range(action_num)[:,None],(1,2))
    batch_size = tf.shape(chol_flat)[0]
    diag_i_batch = tf.concat([
        tf.tile(tf.range(batch_size)[:,None], (1,action_num))[...,None],
        tf.tile(diag_i[None,...],(batch_size, 1, 1))
    ], axis=-1)
    chol_diags = tf.gather_nd(chol_tri, diag_i_batch, name='actor_chol_diag')
    chol_diags = tf.math.softplus(chol_diags, name='actor_softplus')
    sigma_chol = tf.tensor_scatter_nd_update(
        chol_tri, diag_i_batch, chol_diags, name='actor_sigma_chol'
    )



    model = keras.Model(
        inputs=encoder_inputs,
        outputs=[mu, sigma_chol],
        name='actor'
    )

    return model
