# Mahendra, Samarth
# 1001_974_557
# 2023_04_02
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, regularizers

def confusion_matrix(y_true, y_pred, n_classes=10):
    
    temp_mat = np.zeros((n_classes, n_classes), dtype=int)
    for a,b in zip(y_true, y_pred):
        temp_mat[a,b] =temp_mat[a,b] + 1

    return temp_mat



def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):

    tf.keras.utils.set_random_seed(5368) 
    model = models.Sequential()
    model.add(layers.Conv2D(8,(3,3),strides = (1,1),padding = 'same',activation = 'relu',kernel_regularizer=regularizers.l2(0.0001),input_shape= (28,28,1)))
    model.add(layers.Conv2D(16,(3,3),strides = (1,1),padding = 'same',activation = 'relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    model.add(layers.Conv2D(32,(3,3),strides = (1,1),padding = 'same',activation = 'relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Conv2D(64,(3,3),strides = (1,1),padding = 'same',activation = 'relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dense(10, activation='linear',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Activation('softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs,batch_size=batch_size,validation_split=0.2)

    y_pred = np.argmax(model.predict(X_test),axis=1)
    y_true = np.argmax(Y_test,axis=1)

    con_mat = confusion_matrix(y_true, y_pred, n_classes=10)

    model.save('model.h5')

    plt.matshow(con_mat)
    plt.colorbar()
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.savefig('confusion_matrix.png')
    
    return[model,history,con_mat,y_pred]
