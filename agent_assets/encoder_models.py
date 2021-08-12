import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .nn_tools import *
"""
Encoder models encode states into a feature tensor.

Encoder model functions should take a following argument:
    1. observation_space : Dict
Encoder model functions should return:
    1. output tensor
    2. list of Inputs
"""

def encoder_whisker_conv(observation_space):
    obs_inputs = keras.Input(observation_space['obs'].shape, name='obs')
    x = layers.Conv2D(
        32, 
        3, 
        padding='same',
        activation='relu',
        name='encoder_conv1'
    )(obs_inputs)
    x = layers.Conv2D(
        64,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck1'
    )(x)
    x = layers.Conv2D(
        128,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck2'
    )(x)
    x = layers.Conv2D(
        256,
        3,
        padding='same',
        strides=2,
        name='encoder_bottleneck3'
    )(x)
    x = layers.Conv2D(
        512,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck4'
    )(x)
    obs_outputs = layers.Flatten(name='encoder_flatten')(x)

    whi_inputs = keras.Input(observation_space['whi'].shape,name='whi')
    w = layers.Conv1D(
        8,
        2,
        activation='relu',
        name='encoder_conv1d1'
    )(whi_inputs)
    w = layers.Conv1D(
        16,
        2,
        activation='relu',
        name='encoder_conv1d2'
    )(w)
    whi_outputs = layers.Flatten(name='encoder_whi_flatten')(w)

    outputs = layers.Concatenate(name='encoder_output_concat')([
        obs_outputs, whi_outputs
    ])


    return outputs, [obs_inputs, whi_inputs]


def encoder_simple_dense(observation_space):
    inputs = keras.Input(observation_space['obs'].shape,
                         name='obs')
    x = layers.Flatten(name='encoder_flatten')(inputs)
    x = layers.Dense(256, activation='relu',
                         name='encoder_dense1')(x)
    outputs = layers.Dense(256, activation='linear',
                         name='encoder_dense2')(x)
    return outputs, [inputs]

def encoder_mini_dense(observation_space):
    inputs = keras.Input(observation_space['obs'].shape,
                         name='obs')
    x = layers.Flatten(name='encoder_flatten')(inputs)
    x = layers.Dense(64, activation='relu',
                         name='encoder_dense1')(x)
    outputs=x
    return outputs, [inputs]


def encoder_simple_res(observation_space):
    inputs = keras.Input(observation_space['obs'].shape,
                         name='obs')
    x = layers.Conv2D(
        32, 
        3, 
        padding='same',
        activation='relu',
        name='encoder_conv1'
    )(inputs)
    x = res_block(x, 2, name='encoder_resblock1')
    x = layers.Conv2D(
        64,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck1'
    )(x)
    x = res_block(x, 2, name='encoder_resblock2')
    x = layers.Conv2D(
        128,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck2'
    )(x)
    x = res_block(x, 2, name='encoder_resblock3')
    x = layers.Conv2D(
        256,
        3,
        padding='same',
        strides=2,
        name='encoder_bottleneck3'
    )(x)
    outputs = layers.GlobalMaxPool2D(name='encoder_pool',dtype='float32')(x)
    return outputs, [inputs]

def encoder_simple_conv(observation_space):
    inputs = keras.Input(observation_space['obs'].shape,
                         name='obs')
    x = layers.Conv2D(
        32, 
        3, 
        padding='same',
        activation='relu',
        name='encoder_conv1'
    )(inputs)
    x = layers.Conv2D(
        64,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck1'
    )(x)
    x = layers.Conv2D(
        128,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck2'
    )(x)
    x = layers.Conv2D(
        256,
        3,
        padding='same',
        strides=2,
        name='encoder_bottleneck3'
    )(x)
    x = layers.Conv2D(
        512,
        3,
        padding='same',
        strides=2,
        activation='relu',
        name='encoder_bottleneck4'
    )(x)
    x = layers.Flatten(name='encoder_flatten')(x)
    outputs = layers.Dense(
        256,
        activation='linear',
        name='encoder_Dense'
    )(x)
    return outputs, [inputs]
