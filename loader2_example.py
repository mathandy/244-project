from loader2 import load_as_tf_dataset
import tensorflow as tf
tfkl = tf.keras.layers


_DEFAULT_CONV_PARAMS = {
    'activation': 'relu',
    'padding': 'same',
    'kernel_initializer': 'he_normal'
}

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
INPUT_LENGTH = 159744


def get_flat_denoiser():
    model = tf.keras.models.Sequential(layers=[
        tfkl.Lambda(lambda inputs: tf.expand_dims(inputs, -1)),
        tfkl.Conv1D(64, 3, name='den_conv0', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(64, 3, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(64, 3, name='den_conv1', **_DEFAULT_CONV_PARAMS),
        tfkl.Conv1D(1, 3, name='den_conv2', **_DEFAULT_CONV_PARAMS),
        tfkl.Lambda(lambda outputs: tf.squeeze(outputs))
    ])
    return model


if __name__ == '__main__':

    ds_clean, ds_noisy = load_as_tf_dataset()
    ds = tf.data.Dataset.zip((ds_noisy, ds_clean))
    ds = ds.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    # create model
    model = get_flat_denoiser()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    loss_metric = tf.keras.metrics.Mean()
    loss_fcn = tf.keras.losses.MSE

    # train
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,))
        for step, (x_batch, y_batch) in enumerate(ds):
            with tf.GradientTape() as tape:
                yhat_batch = model(x_batch)
                loss = loss_fcn(x_batch, yhat_batch)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
