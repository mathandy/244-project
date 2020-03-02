import tensorflow as tf
tfkl = tf.keras.layers


def get_generator(input_static_dim=59,
                  input_dynamic_dim=118,
                  num_hidden=5,
                  hidden_dim=512,
                  dropout_rate=0.5):
    """Credit: from MIT Licensed
        https://github.com/tkm2261/dnn-voice-changer
    """

    inputs_all = x = tfkl.Input(shape=(input_static_dim + input_dynamic_dim,))
    x_static = tfkl.Lambda(lambda x: x[:, :input_static_dim])(inputs_all)

    t_x = tfkl.Dense(input_static_dim, activation='sigmoid')(x_static)

    for _ in range(num_hidden):
        act = tfkl.LeakyReLU()
        x = tfkl.Dense(hidden_dim, activation=act)(x)
        x = tfkl.Dropout(dropout_rate)(x)

    g_x = tfkl.Dense(input_static_dim)(x)

    outputs = tfkl.add([x_static, tfkl.multiply([t_x, g_x])])

    model = tf.keras.Model(inputs=inputs_all, outputs=outputs, name='generator')

    return model