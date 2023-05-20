"""
Module: utils.py
contains utility function definitions that will be imported and used in the notebook(.ipynb) file  

"""


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import InputLayer
import time

def generate_random_dnn_config(layers, how_many):
  """
    Return a 2d numpy.array each row of which represents a deep FCNN architecture encoding
    Each row is a 1D vector each element of which is the number of neurons(nodes) on the corresponding layer in the NN
    (for ex: X[0,0]  shows how many neurons are in the first layer of the first generated architecture encoding)

    ** Note : not all config vectors have to be filled with all non-zero values, some architectures can have only input and output layers or input,a few hidden layers and output layer, 
     and the rest will be 0s indicating 0 neurons on those layers, i.e. no need to create those layers later in ***create_mode_from_dnn_config*
    

  """
  X = np.zeros((how_many, layers), dtype= np.int32)

  for i in range(X.shape[0]):
      n_of_empty_layers = np.random.randint(0,layers-1 )  #shows how many elements in the ending of the current config vector should be 0s(0 neurons)
      n_of_full_layers = layers - n_of_empty_layers   # other layers should be filled with random non-zero numbers(full layers)
      x = np.random.randint(1, 50, size=n_of_full_layers - 2)  #the #of nodes can range from 1 to 50(just a predefined rule, no reason behind that)
      X[i,1:x.shape[0]+1] = x[:,np.newaxis].T  # set the #of nodes of the hidden layers(skipping the 0th input layer) to the generated nodes
      num = np.random.randint(1,50)  
      X[i,0] = num  # both the input and the output layers should have the same number of neurons
    #   (since these architectures take a signal of some length and just output a random signal vector of the same length)- imitation of 1D signal processing
      X[i,len(x)+1] = num

  return X


def create_model_from_dnn_config(config):

  """
    Create and return a tensorflow.keras Sequential model (deep Fully-connected neural network(FCNN)) based on the given architecture config vector
    Each of these models depends on the default keras's weights and params, no training or tuning will be done. The returned model is just an 
    imitation of a seq2seq(sequence-to-sequence) model that has no activation function in the end, and its output won't be used anywhere. 
    This returned model will be the one for which the latency will be estimated
  """

  lays =[]
  lays.append(layers.InputLayer(input_shape=(config[0], ))) # add as many neurons as many there are on the first cell of the config vector(the input layer)
  for i in range(1, len(config)):
    if ((i == len(config) - 1) or (config[i+1] == 0)): # if we are not at the last layer and the next one is empty,
      # just create the current as an output layer without any activation function and exit
      lays.append(layers.Dense(config[i])) 
      break
    lays.append(layers.Dense(config[i], activation="relu")) #otherwise, if it is a hidden layer, use the non-linear relu activ. function

  model = keras.Sequential(lays) #create and return the obtained keras model
  return model


def create_signal_data(input_shape, length = 50):
  """
      Create and return a list of lists, each of which represents a portion of the generated 1D signal data.
      The 1D data of default length 50( filled with random integers)  will be splitted into as many parts as needed for each of them
      (called batches) to be inputted to a seq2seq model's input layer. 
      So all subsplits(batches) of the 1D vector data (which is a proxy for signal or time-series data)
      should be of the same size as the number of nodes ont he input layer
      and if the last batch can not be fully feeded, we fill the rest of the batch with 0s, meaning missing inputs for those ending nodes.

    """

  batches = list()
  for i in range(length):   
    batches.append(np.random.randint(0,100000)) 
  
  result = list()
  sublist_size = length // input_shape
  remainder = length % input_shape 

  start = 0
  end = input_shape

  for i in range(sublist_size): #for each of the fullly populated batches
    result.append(batches[start : end]) # take the corresponding splitted part from the generated 1D signal data
    start = end
    end += input_shape

  if (remainder != 0): # if the last batch can not perfectly be populated
    last_batch = [0 for _ in range(input_shape)] # add from the beginning the remaining datapoints from the generated 1D signal data
    last_batch[0 : remainder] = batches[start : ] # and fill the missing ones with zeros
    result.append(last_batch)

  return result #return the created batches


def measure_dnn_latency(config):
  """
    Take one configuration vector standing for a single seq2seq input architecture and return the average latency(measured and averaged for 5 times)
    This measured avg latency will be used later as the ground truth*target variable) in creating the dataset.
  """

  model = create_model_from_dnn_config(config) 
  input_data = create_signal_data(config[0])

  model.compile(loss="mse", metrics=["mse"], optimizer = "Adam")  #compile the input architecture whose latency is to be measured with the following hyperparams

  latencies = list()
  for  i in range(5): 

      t0 = time.time()
      y = model.predict(input_data) #we don't call model.fit() because there is no need to improve the by default generated tf.keras model's weights,
      # we just want to estimate the inference(.predict() time) based on already initialized random weights,
      # input_data is the 1D data, so with one line all batches are fed to the model

      latency = time.time() - t0 #record the lasted latency for feeding all the batches of one single signal data
      latencies.append(latency)
  result = sum(latencies)/ len(latencies) # average of all the recorded 5 latencies 
  return result # output the avg. latency for the inference stage(prediction) of the model


def create_dataset(n_layers,n_models):
  """
    Create and save the dataset containing the encoded input architectures and their corresponding estimated avg. latencies
  """
  X = generate_random_dnn_config(n_layers, n_models)
  y = list()

  for architecture in X:
    y.append(measure_dnn_latency(architecture))

  y = pd.Series(y)
  df = pd.DataFrame(X)

  col_names = []
  for i in range(n_layers):
      col_names.append(("Layer_" + str(i+1), df[i]))
  col_names.append(("Latency", y))

  # Concatenate the DataFrames and Series
  df = pd.concat([df for _, df in col_names], axis=1, keys=[key for key, _ in col_names]) # add the descriptive col names, 
  #such as Layer_1,...and the last col is the groundtruths "latency"
  df.to_csv('/content/dataset.csv', index=False) # save the data as a .csv file in the given filepath