# Mahendra, Samarth
# 1001_974_557
# 2023_04_16
# Assignment_04_01
 
import tensorflow as tf
import numpy as np
import keras
from keras import layers,models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class CNN(object):
    def __init__(self):
        self.model = keras.Sequential()
        self.metrics = []

    def add_input_layer(self, shape=(2,),name="" ):
        self.input_dimensions = shape
        self.model.add(layers.InputLayer(input_shape = shape,name = name))
        return None   

    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        self.model.add(layers.Dense(units = num_nodes,activation = activation,name = name, trainable = trainable))
        return None

    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        return self.model.add(layers.Conv2D(filters = num_of_filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation, name=name, trainable=trainable))

    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        return self.model.add(layers.MaxPooling2D(pool_size=pool_size, padding=padding, strides=strides, name=name)) 

    def append_flatten_layer(self,name=""):
        return self.model.add(keras.layers.Flatten())

    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        if not layer_numbers:
            layer_numbers = [i for i, layer in enumerate(self.model.layers) if layer.name in layer_names]
        else:
            layer_numbers = [num - 1 for num in layer_numbers]

        for i in layer_numbers:
            self.model.layers[i].trainable = trainable_flag
        return None  

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        if not layer_number:
            for i in range(len(self.model.layers)):
                    if self.model.layers[i].name==layer_name:
                        weights = self.model.layers[i].get_weights()
                        if weights ==[]:
                            return None
                        return weights[0]
        else:
            if layer_number>=0:
                weights = self.model.layers[layer_number-1].get_weights()
            else:
                weights = self.model.layers[layer_number].get_weights()
            if weights == []:
                return None
            return weights[0]

    def get_biases(self,layer_number=None,layer_name=""):
        if not layer_number:
            for i in range(len(self.model.layers)):
                    if self.model.layers[i].name==layer_name:
                        weights = self.model.layers[i].get_weights()
                        if weights ==[]:
                            return None
                        return weights[1]
        else:
            weights = self.model.layers[layer_number-1].get_weights()
            if weights ==[]:
                return None
            return weights[1]

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        if not layer_number:
            for i in range(len(self.model.layers)):
                    if self.model.layers[i].name==layer_name:
                        w =  self.model.layers[i].get_weights()[1]
                        self.model.layers[i].set_weights([weights,w])
        else:
            w =  self.model.layers[layer_number-1].get_weights()[1]
            self.model.layers[layer_number-1].set_weights([weights,w])

    def set_biases(self,biases,layer_number=None,layer_name=""):
        if not layer_number:
            for i in range(len(self.model.layers)):
                    if self.model.layers[i].name==layer_name:
                        w =  self.model.layers[i].get_weights()[0]
                        self.model.layers[i].set_weights([w,biases])
        else:
            w =  self.model.layers[layer_number-1].get_weights()[0]
            self.model.layers[layer_number-1].set_weights([w,biases])

    def remove_last_layer(self):
        last = self.model.pop()
        self.model = keras.Sequential(self.model.layers)
        return last

    def load_a_model(self,model_name="",model_file_name=""):
        if not model_name:
            self.model = keras.models.load_model(model_file_name)
        else:
            if model_name.lower() == "vgg19":
                self.model = keras.Sequential(layers = keras.applications.vgg19.VGG19().layers)
            elif model_name.lower() == "vgg16":
                self.model = keras.Sequential(layers = keras.applications.vgg16.VGG16().layers)
        return self.model  

    def save_model(self,model_file_name=""):
       return self.model.save(model_file_name)

    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        self.loss = loss
        return None

    def set_metric(self,metric):
        self.metric = metric
        return None

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        if optimizer == "SGD":
            self.optimizer=keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
        elif optimizer =="RMSprop":
            self.optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == "Adagrad":
            self.optimizer=keras.optimizers.Adagrad(learning_rate=learning_rate)
        return None

    def predict(self, X):
        return self.model.predict(X.astype('float32'))

    def evaluate(self,X,y):
        test_loss, test_metrics = self.model.evaluate(X,y)
        return test_loss, test_metrics

    def train(self, X_train, y_train, batch_size, num_epochs):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metric)
        history = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs)
        return history.history['loss']
    