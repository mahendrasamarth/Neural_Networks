# Mahendra, Samarth
# 1001_974_557
# 2023_04_16
# Assignment_04_02

import pytest
import numpy as np
from Mahendra_04_01 import CNN
import os
import tensorflow as tf
import keras

def test_train():
    from keras.datasets import cifar10
    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    number_of_train_samples_to_use = 100
    X_train = X_train[0:number_of_train_samples_to_use, :]
    y_train = y_train[0:number_of_train_samples_to_use]
    my_cnn = CNN()
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=3,padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.set_metric('mse')
    my_cnn.set_loss_function('MeanSquaredError')
    my_cnn.set_optimizer(optimizer="SGD",learning_rate=0.01,momentum=0.0)
    out = my_cnn.train(X_train, y_train, batch_size,epochs)
    assert len(out) == epochs


def test_decreasing_loss():
    from keras.datasets import cifar10
    batch_size = 32
    num_classes = 10
    epochs = 10
    data_augmentation = True
    num_predictions = 20
    keras.utils.set_random_seed(1234)
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    number_of_train_samples_to_use = 100
    X_train = X_train[0:number_of_train_samples_to_use, :]
    y_train = y_train[0:number_of_train_samples_to_use]
    my_cnn=CNN()
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=3,padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.set_metric('mse')
    my_cnn.set_loss_function('MeanSquaredError')
    my_cnn.set_optimizer(optimizer="SGD",learning_rate=0.01,momentum=0.0)
    out = my_cnn.train(X_train, y_train, batch_size,epochs)
    assert out[9]<out[0]