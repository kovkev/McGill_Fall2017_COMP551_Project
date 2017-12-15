import keras
from keras.datasets import mnist, cifar10, cifar100
from keras import backend as K

def get_mnist_data():
    data = {
        "output_shape": 10,
    }

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_rows = 28
    img_cols = 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        # x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        # x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        # input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        # input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, data["output_shape"])
    y_test = keras.utils.to_categorical(y_test, data["output_shape"])

    data["x_train"] = x_train
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["input_shape"] = x_train[0].shape
    data["output_shape"] = y_train[0].shape

    return data


def get_cifar10_data():
    data = {
        "output_shape": 10,
    }

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, data["output_shape"])
    y_test = keras.utils.to_categorical(y_test, data["output_shape"])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    data["x_train"] = x_train
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["input_shape"] = x_train[0].shape
    data["output_shape"] = y_train[0].shape

    return data

def get_cifar100_data():
    data = {
        "output_shape": 100,
    }

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, data["output_shape"])
    y_test = keras.utils.to_categorical(y_test, data["output_shape"])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    data["x_train"] = x_train
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["input_shape"] = x_train[0].shape
    data["output_shape"] = y_train[0].shape

    return data

