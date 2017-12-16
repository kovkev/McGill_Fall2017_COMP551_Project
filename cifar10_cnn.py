'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os
import re
import numpy as np

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
steps_per_epoch = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

from paper import get_m_r


def create_hidden_layer(
        budget,
        layer_index,
        data_shape,
):
    layer_size = int(get_m_r(budget, size(data_shape["input_shape"]), size(data_shape["output_shape"]), layer_index))
    layer = Dense(
        layer_size,
    )
    return layer


def size(t):
    out = 1
    for d in t:
        out *= d
    return out

def generate_model(
        task_name,
        data_shape,
        num_hidden_layers,
        budget,
        activation,
        dirdir,
):
    model = Sequential()

    num_params_in_layer = int(get_m_r(budget, size(data_shape["input_shape"]), size(data_shape["output_shape"]), 0))
    model.add(Dense(
        num_params_in_layer,
        input_shape = data_shape["input_shape"]
    ))

    for i in range(num_hidden_layers):
        layer = create_hidden_layer(
            budget,
            i,
            data_shape,
        )
        model.add(layer)
        model.add(Activation(activation))

    last_layer = num_classes
    last_layer = data_shape["y_train_size"]
    model.add(Dense(last_layer))
    model.add(Activation('softmax'))

    start_epoch = try_to_load_weights(
        task_name,
        model,
        dirdir,
    )

    return model, start_epoch

def try_to_load_weights(
        task_name,
        model,
        dirdir,
):
    path = ["weights", task_name, dirdir]
    from os import listdir
    from os.path import isfile, join
    weight_files = [f for f in listdir(join(*path)) if isfile(join(*(path + [f])))]
    weight_files = list(reversed(sorted(weight_files)))

    if len(weight_files) > 0:
        last_weight_file = weight_files[0]
        epochs_done = int(re.findall(r"\d+", last_weight_file)[0])
        model.load_weights(os.path.join(*[
            "weights",
            task_name,
            dirdir,
            last_weight_file,
        ]))

        return epochs_done

    return 0


def directory(
        task_name,
        budget,
        num_hidden_layers,
        activation_function,
        loss_function,
):
    if type(activation_function).__name__ == "function":
        activation_function = activation_function.__name__

    out = "_".join([
        str(task_name),
        str(budget),
        "k",
        str(num_hidden_layers),
        str(activation_function),
        str(loss_function),
    ])

    return out

def makedir(l):
    if not os.path.exists(os.path.join(*l)):
        os.makedirs(
            os.path.join(*l)
        )

def test_task(
        task_name,
        data,
        budget,
        num_hidden_layers,
        activation_function,
        loss_function,
        data_augmentation,
):
    dirdir = directory(
        task_name,
        budget,
        num_hidden_layers,
        activation_function,
        loss_function,
    )

    makedir([
        "weights",
        task_name,
        dirdir,
    ])
    makedir([
        "metrics",
        task_name,
        dirdir,
    ])

    model, epochs_done = generate_model(
        task_name,
        {
            "input_shape": data["input_shape"],
            "output_shape": data["output_shape"],
            "y_train_size": data["y_train"][0].size,
        },
        num_hidden_layers,
        budget,
        activation_function,
        dirdir,
    )

    # initiate RMSprop optimizer
    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    opt = keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0
    )

    # Let's train the model using RMSprop
    model.compile(loss=loss_function,
                optimizer=opt,
                metrics=['accuracy'])

    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    fullpath = os.path.join(*["weights", task_name, dirdir, filepath])
    mcp = ModelCheckpoint(
        fullpath,
        monitor='val_acc',
        verbose=1,
        save_best_only=False,
    )

    class LossHistory(keras.callbacks.Callback):
        def __init__(self, fullpath):
            self.fullpath = fullpath
            super(LossHistory, self)

        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            return
            import pdb; pdb.set_trace()
            self.losses.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            import json

            fullpath = ["metrics", task_name, dirdir, "epoch-%s.json" % (str(epoch + 1)) ]

            with open(os.path.join(*fullpath), 'w') as f:
                f.write(json.dumps(logs))

    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    fullpath = os.path.join(*["metrics", task_name, filepath])
    history = LossHistory(
        fullpath,
    )

    callbacks_list = [mcp, history]

    # import pdb; pdb.set_trace()
    if not data_augmentation:
        # import pdb; pdb.set_trace()
        # for i in range(epochs_done, epochs):

        print('Not using data augmentation.')
        def fitGenerator():

            while 1:
                how_many_train = 50
                how_many_test = 50
                num_train = data["x_train"].shape[0]
                num_test = data["x_test"].shape[0]
                nums_to_train = np.arange(num_train)
                nums_to_test = np.arange(num_test)
                np.random.shuffle(nums_to_train)
                np.random.shuffle(nums_to_test)
                nums_to_train = nums_to_train[:how_many_train]
                nums_to_test = nums_to_test[:how_many_test]

                yield data["x_train"][nums_to_train], data["y_train"][nums_to_train]

        model.fit_generator(
            fitGenerator(),
            # data["x_train"][nums_to_train],
            # data["y_train"][nums_to_train],
            # batch_size=batch_size,
            epochs=epochs,
            samples_per_epoch=100,
            verbose=2,
            initial_epoch=min(epochs_done, 95),
            validation_data=(
                data["x_test"][:5000],
                data["y_test"][:5000],
            ),
            callbacks=callbacks_list,
        )
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)


        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=(x_test, y_test),
                            workers=4)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(
        data["x_test"],
        data["y_test"],
        verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
