import tensorflow as tf
from itertools import islice
import time


def run(model, train_ds, valid_ds, n_iter=None, n_epoch=1, lr=0.01, momentum=0.9):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    if model is not None:
        model.compile(optimizer, tf.keras.losses.SparseCategoricalCrossentropy(True))

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = model.loss(y, prediction)
        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, tf.argmax(prediction, 1)

    @tf.function
    def valid_step(x, y):
        prediction = model(x, training=False)
        loss = model.loss(y, prediction)
        return loss, tf.argmax(prediction, 1)

    if model is None:
        @tf.function
        def train_step(x, y):
            return 0., y

        @tf.function
        def valid_step(x, y):
            return 0., y

    def train_epoch(data, n_iter=None):
        loss_metric = tf.keras.metrics.Mean()
        accu_metric = tf.keras.metrics.Accuracy()
        iterator = data if n_iter is None else islice(data, n_iter)
        for x, y in iterator:
            loss, pred = train_step(x, y)
            accu_metric.update_state(y, pred)
            loss_metric.update_state(loss)
        return loss_metric.result(), accu_metric.result()

    def valid_epoch(data, n_iter=None):
        loss_metric = tf.keras.metrics.Mean()
        accu_metric = tf.keras.metrics.Accuracy()
        iterator = data if n_iter is None else islice(data, n_iter)
        for x, y in iterator:
            loss, pred = valid_step(x, y)
            accu_metric.update_state(y, pred)
            loss_metric.update_state(loss)
        return loss_metric.result(), accu_metric.result()

    # WARMUP
    warmup_iter = min(5, n_iter)
    train_epoch(train_ds, n_iter=warmup_iter)
    if valid_ds:
        valid_epoch(valid_ds, n_iter=warmup_iter)
    train_time_data = []
    valid_time_data = []

    for epoch_idx in range(n_epoch):
        t0 = time.time()
        train_loss, train_accu = train_epoch(train_ds, n_iter=n_iter)
        dt = time.time() - t0
        train_time_data.append(dt)

        print(
            f"{epoch_idx:<3}",
            f"tr loss: {train_loss:6.3f}",
            f"tr accu: {train_accu:6.3f}",
            f"tr time: {dt: 5.2f}",
            sep=" | ",
            end="",
        )
        if valid_ds:
            t0 = time.time()
            valid_loss, valid_accu = valid_epoch(valid_ds, n_iter=n_iter)
            dt = time.time() - t0
            valid_time_data.append(dt)

            print(
                f" | "
                f"vd loss: {valid_loss:6.3f}",
                f"vd accu: {valid_accu:6.3f}",
                f"vd time: {dt: 5.2f}",
                sep=" | ",
                end="",
            )
        print()
    return train_time_data, valid_time_data
