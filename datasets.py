import tensorflow as tf


def get_mnist(train_bs, valid_bs, prefetch=False):
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')[..., None] / 255.
    x_valid = x_valid.astype('float32')[..., None] / 255.

    tds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    vds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    def train_prep(x, y):
        x = tf.pad(x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]])
        x = tf.image.random_crop(x, size=(x.shape[0], 28, 28, 1))
        return x, y

    tds = tds.shuffle(10000)
    tds = tds.batch(train_bs, drop_remainder=True)
    tds = tds.map(train_prep)
    vds = vds.batch(valid_bs, drop_remainder=True)
    if prefetch:
        tds = tds.prefetch(tf.data.AUTOTUNE)
        vds = vds.prefetch(tf.data.AUTOTUNE)
    return tds, vds, (28, 28, 1), 10


def get_cifar(train_bs, valid_bs, prefetch=False):
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_valid = x_valid.astype('float32') / 255.

    tds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    vds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    def train_prep(x, y):
        x = tf.pad(x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
        x = tf.image.random_crop(x, size=(train_bs, 32, 32, 3))
        x = tf.image.random_flip_left_right(x)
        return x, y

    tds = tds.shuffle(10000).batch(train_bs, drop_remainder=True).map(train_prep)
    vds = vds.batch(valid_bs, drop_remainder=True)
    if prefetch:
        tds = tds.prefetch(tf.data.AUTOTUNE)
        vds = vds.prefetch(tf.data.AUTOTUNE)
    return tds, vds, (32, 32, 3), 10
