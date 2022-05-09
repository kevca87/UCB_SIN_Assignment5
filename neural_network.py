import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#This is the base class
class NeuralNetwork(tf.keras.Model):
  def __init__(self,inputs_n):
    super(NeuralNetwork, self).__init__()
    self.sequence = keras.models.Sequential()
    self.sequence.add(keras.layers.Input(shape=(inputs_n,)))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime

#This class is implemented when we want to work with one activation function
class NeuralNetworkOneActFn(NeuralNetwork):

  def __init__(self,inputs_n,hidden_layers_sizes,activation_fn):
    NeuralNetwork.__init__(self,inputs_n)
    for i in hidden_layers_sizes:
        self.sequence.add(tf.keras.layers.Dense(i, activation=activation_fn))
    self.sequence.add(tf.keras.layers.Dense(3,activation='softmax'))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime

#This class supports two activation functions
class NeuralNetworkTwoActFn(NeuralNetwork):

  def __init__(self,inputs_n,hidden_layers_sizes,activation_fn,sec_act_fn):
    NeuralNetwork.__init__(self,inputs_n)
    for i in hidden_layers_sizes:
        self.sequence.add(tf.keras.layers.Dense(i, activation=activation_fn))
        self.sequence.add(tf.keras.layers.Dense(i, activation=sec_act_fn))
    self.sequence.add(tf.keras.layers.Dense(3,activation='softmax'))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime

#This class implements the dropout layer between the hidden layers 
class NeuralNetworkDropout(NeuralNetwork):

  def __init__(self,inputs_n,hidden_layers_sizes,activation_fn,dropout_rate=0.5):
    NeuralNetwork.__init__(self,inputs_n)
    for i in hidden_layers_sizes:
        self.sequence.add(tf.keras.layers.Dense(i, activation=activation_fn))
        self.sequence.add(tf.keras.layers.Dropout(dropout_rate))
    self.sequence.add(tf.keras.layers.Dense(3,activation='softmax'))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime