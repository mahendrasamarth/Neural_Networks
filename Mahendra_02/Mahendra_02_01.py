# Mahendra, Mahendra
# 1001_974_557
# 2023_03_19
# Assignment_02_01

import numpy as np
import tensorflow as tf


def generate_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    # if there's any data left, yield it
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]



def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                              validation_split=[0.8,1.0],weights=None,seed=2):
  final_batch_output = []

  tf_weights = []
  epoch_loss = []
  y_pred = []
  X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), 1)
  X_train, Y_train, X_val, y_val = split_data(X_train, Y_train, validation_split)
  X_train = tf.convert_to_tensor(X_train,dtype=tf.float32)
  Y_train = tf.convert_to_tensor(Y_train,dtype=tf.float32)
  X_val = tf.convert_to_tensor(X_val,dtype=tf.float32)
  y_val = tf.convert_to_tensor(y_val,dtype=tf.float32)
  if weights == None:
    weights = []
    np.random.seed(seed)
    weights.append(tf.Variable((np.random.randn(X_train.shape[1],layers[0])).astype(np.float32)))
    for i in range(1, len(layers)):
      np.random.seed(seed)
      weights.append(tf.Variable((np.random.randn(layers[i-1]+1,layers[i])).astype(np.float32)))
  else:
    weights = weights

  for e in range(epochs):

    for k in range(0,X_train.shape[0],batch_size):
      outlayer =[]
      out_val=[]
      temp_xbatch = X_train[k:k+batch_size]
      temp_ybatch = Y_train[k:k+batch_size]
      with tf.GradientTape() as tape:
        tape.watch(weights)
        outlayer.append(tf.matmul(temp_xbatch,weights[0]))
        outlayer[0] = activation_functions(activations[0],outlayer[0])
        for j in range(1,len(layers)):
          x = tf.concat([tf.ones((tf.shape(outlayer[j-1])[0], 1)), outlayer[j-1]], axis=1)
          outlayer.append((tf.matmul(x,weights[j])))
          outlayer[j] = activation_functions(activations[j],outlayer[j])
        y_pred = outlayer[-1]
        loss_value = loss_functions(loss,y_pred,temp_ybatch)
      gradients = tape.gradient(loss_value,weights)
      for l in range(len(weights)):
        weights[l] = weights[l] - alpha*gradients[l]
    if X_train.shape[0] % batch_size != 0:
      outlayer =[]
      out_val=[]
      temp_xbatch = X_train[-(X_train.shape[0] % batch_size):]
      temp_ybatch = Y_train[-(X_train.shape[0] % batch_size):]
      with tf.GradientTape() as tape:
        tape.watch(weights)
        outlayer.append(tf.matmul(temp_xbatch,weights[0]))
        outlayer[0] = activation_functions(activations[0],outlayer[0])
        for j in range(1,len(layers)):
          x = tf.concat([tf.ones((tf.shape(outlayer[j-1])[0], 1)), outlayer[j-1]], axis=1)
          outlayer.append((tf.matmul(x,weights[j])))
          outlayer[j] = activation_functions(activations[j],outlayer[j])
        y_pred = outlayer[-1]
        loss_value = loss_functions(loss,y_pred,temp_ybatch)
      gradients = tape.gradient(loss_value,weights)
      for l in range(len(weights)):
        weights[l] = weights[l] - alpha*gradients[l]
    # print(f'Updated weights{weights}')


    out_val.append(tf.matmul(X_val,weights[0]))
    out_val[0] = activation_functions(activations[0],out_val[0])
    for j in range(1,len(layers)):
        x = tf.concat([tf.ones((tf.shape(out_val[j-1])[0], 1)), out_val[j-1]], axis=1)
        out_val.append((tf.matmul(x,weights[j])))
        out_val[j] = activation_functions(activations[j],out_val[j])
    y_pred = out_val[-1]
    epoch_loss.append(loss_functions(loss,y_pred,y_val))
  out_val=[]
  out_val.append(tf.matmul(X_val,weights[0]))
  out_val[0] = activation_functions(activations[0],out_val[0])
  for j in range(1,len(layers)):
      x = tf.concat([tf.ones((tf.shape(out_val[j-1])[0], 1)), out_val[j-1]], axis=1)
      out_val.append((tf.matmul(x,weights[j])))
      out_val[j] = activation_functions(activations[j],out_val[j])
  out = out_val[-1]
  return [weights,epoch_loss, out]

def split_data(X_train, Y_train, validation_split):
    start = int(validation_split[0] * X_train.shape[0])
    end = int(validation_split[1] * X_train.shape[0])
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate(
        (Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]




def activation_functions(activations,layer_output):
  if activations.lower() == 'sigmoid':
    return tf.nn.sigmoid(layer_output)

  elif activations.lower() == 'linear':
    return layer_output

  elif activations.lower() == 'relu':
    return tf.nn.relu(layer_output)
  

#batches
def generate_batches(X_train, Y_train, batch_size):
    for i in range(0, X_train.shape[0], batch_size):
        yield X_train[i:i+batch_size], Y_train[i:i+batch_size]
    # if there's any data left, yield it
    if X_train.shape[0] % batch_size != 0:
        yield X_train[-(X_train.shape[0] % batch_size):], Y_train[-(X_train.shape[0] % batch_size):]



def loss_functions(loss,y_pred,temp_ybatch):

  if loss.lower() == 'mse':
    loss_mse = tf.reduce_mean(tf.square(temp_ybatch - y_pred))
    return loss_mse

  elif loss.lower() == 'svm':
    loss_svm = tf.reduce_mean(tf.maximum(0.0,1.0-temp_ybatch*y_pred))
    return loss_svm

  elif loss.lower() == 'cross_entropy':
    loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = temp_ybatch ))
    return loss_cross_entropy
  else:
     pass