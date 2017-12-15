# Our testing

import cifar10_cnn

import data

import paper

from keras import backend as K
def squared(x):
    return x * x
# from keras.utils.generic_utils import get_custom_objects
# get_custom_objects().update({'custom_activation': Activation(custom_activation)})

ks = [0,2,4,8]
activation_functions = [
    'linear',
    'tanh',
    'relu',
    squared,
]
data_augmentations = [
    False,
]
loss_functions = [
    'categorical_crossentropy',
]
budgets = [
    500000, # real params
]

tasks = {
    "mnist": data.get_mnist_data,
    "cifar10": data.get_cifar10_data,
    "cifar100": data.get_cifar100_data,
}

import itertools
products = itertools.product(
    budgets,
    ks,
    activation_functions,
    loss_functions,
    data_augmentations,
)

for task_name, get_task_data in tasks.items():
    print(get_task_data)
    data = get_task_data()

    for product in products:
        cifar10_cnn.test_task(
            task_name,
            data,
            *product
        )
