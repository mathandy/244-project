from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras import Model, Sequential
import tensorflow as tf


def build_model(l2_strength, numFeatures, numSegments, learning_rate, beta_1, beta_2, epsilon):

    inputs = Input(shape=[numFeatures, numSegments, 1])
    x = inputs

    # -----
    x = tf.keras.layers.ZeroPadding2D(((4, 4), (0, 0)))(x)
    x = Conv2D(filters=18, kernel_size=[9, 8], strides=[1, 1], padding='valid', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    skip0 = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(skip0)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # -----
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    skip1 = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(skip1)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = x + skip1
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # ----
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = x + skip0
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # ----
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding='same')(x)

    model = Model(inputs=inputs, outputs=x)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)
    # optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=3e-4)

    model.compile(optimizer=optimizer, loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])
    return model

def simple_denoiser(numFeatures, numSegments):
    inputs = Input(shape=[numFeatures, numSegments, 1])
    x = inputs

    # -----
    x = Conv2D(filters=3, kernel_size=[1, 8], strides=[1, 1], padding='valid', use_bias=False,)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False,)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    model = Model(inputs=inputs, outputs=x)

    optimizer = tf.keras.optimizers.Adam(1e-4, 0.9, 0.999, 0)
    # optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=3e-4)

    model.compile(optimizer=optimizer, loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])
    return model

