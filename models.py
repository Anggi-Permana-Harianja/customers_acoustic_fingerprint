'''
these codes are the models used for training
we will use CNN as our network with detailed architecture such as:
            fingerprint input
                  v
                conv2D <- weights
                  v
                bias   <- bias
                  v
                relu
                  v
                maxpool
                  v
                conv2D <- weights
                  v
                bias   <- bias
                  v
                relu
                  v
                maxpool
                  v
                matmul <- FCNN/weights
                  v
                bias   <- bias
                  v
'''

#import modules
import math
import tensorflow as tf

def _next_power_of_two(x):
  '''
  - calculates the next smallest enclosing power of two for an input given
  - args:
    - x: input
  retuns:
    - next largest power of two
  '''
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()

def prepare_model_settings(label_count, sample_rate, clip_duration_ms, window_size_ms, 
                           window_stride_ms, feature_bin_count):
  '''
  - calculate the common settings for model that will be used later
  - args:
    - label_count: how many classes needs to be recognized
    - sample_rate: the rate/number of audio sample per second
    - clip_duration_ms: length of each audio clip
    - window_size_ms: duration freaquency windon
    - window_stride_ms: the duration of striding window
    - feature_bin_count: the number of the bin/band will be used to create fingerprint
  - returns:
    - dictionaro of common settings
  '''
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length _minus_window = (desired_samples - window_stride_samples)

  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)

  average_window_width = -1
  fingerprint_width = feature_bin_count
  fingerprint_size = fingerprint_width * spectrogram_length

  return {
    'desired_samples': desired_samples, 
    'window_size_samples': window_size_samples, 
    'window_stride_samples': window_stride_samples,
    'spectrogram_length': spectrogram_length, 
    'fingerprint_width': fingerprint_width, 
    'fingerprint_size': fingerprint_size, 
    'label_count': label_count, 
    'sample_rate': sample_rate,
    'average_window_width': average_window_width
  }
  

def load_variables_from_checkpoint(sess, start_checkpoint):
  '''
  - checkpoint restoration
  '''
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

def conv_model(fingerprint_input, model_settings, is_training):
  '''
  - create convolution NN model that will be used for both training and testing
  - args:
    - fingerprint_input: tensor node as our input
    - model_settings: dictionary of common settings
    - is_training: yes for trianing, no for testing
  -returns:
    - tensor node of logits result and dropout
  '''
  # dropout_prob only for training and not testing
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name = 'dropout_prob')
  input_frequency = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']

  #--------------------------------------------
  # reshape the fingerprint_input into 4D tensor
  fingerprint_4d = tf.reshape(fingerprint_input, 
                              [-1, input_time_size, input_frequency_size, 1])
  #--------------------------------------------

  #--------------------------------------------
  # create ensamble model that we will use for modelling
  # crete model first layer
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_strides = [1, 1, 1, 1]
  first_padding = 'SAME'

  first_maxpool_stride = [1, 2, 2, 1]
  first_maxpool_ksize = [1, 2, 2, 1]
  first_maxpool_padding = 'SAME'

  first_weights = tf.get_variable(name = 'first_weights', 
                                  initializer = tf.truncated_normal_initializer(stddev = 0.01), 
                                  shape = [first_filter_height, first_filter_width, 1, 
                                          filter_filter_count])
  first_bias = tf.get_variable(name = 'first_bias', 
                               initializer = tf.zeros_initializer, 
                               shape = [first_filter_count])
  # create first conv layer
  first_conv_layer = tf.nn.conv2d(fingerprint_4d, first_weights, first_strides, 
                                  'SAME') + first_bias
  # relu activation function
  first_relu = tf.nn.relu(first_conv_layer)

  #if it is training or testing
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  #maxpooling
  first_maxpool = tf.nn.maxpool(first_dropout, first_maxpool_ksize, first_maxpool_stride, first_maxpool_padding)

  # create conv model second layer
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_filter_strides = [1, 1, 1, 1]
  second_filter_padding = 'SAME'

  second_weights = tf.get_variable(name = 'second_weights', 
                                   initializer = tf.truncated_normal_initializer(stddev = 0.01), 
                                   shape = [second_filter_height, second_filter_width, 
                                            first_filter_count, second_filter_count])
  second_bias = tf.get_variable(name = 'second_bias', 
                                initializer = tf.zeros_initializer, 
                                shape = [second_filter_count])
  # create second conv layer
  second_conv = tf.nn.conv2d(first_maxpool, second_weights, second_filter_strides, 
                             second_filter_padding) + second_bias
  # second layer relu 
  second_relu = tf.nn.relu(second_conv)

  #if it is training or testing
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu

  #get the convolution shape before flatten it
  second_conv_shape = second_conv.get_shape()
  second_conv_output_height = second_conv_shape[1]
  second_conv_output_width = second_conv_shape[2]
  second_conv_element_count = int(second_conv_output_height * second_conv_output_width * second_filter_count)

  flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count]) #vectorize

  label_count = model_settings['label_count']

  final_fcnn_weights = tf.get_variable(name = 'final_fcnn_weights', 
                                       initializer = tf.truncated_normal_initializer(stddev = 0.01), 
                                       shape = [second_conv_element_count, label_count])
  final_fcnn_bias = tf.get_variable(name = 'final_fcnn_bias', 
                                    initializer = tf.zeros_initializer, 
                                    shape = [label_count])

  #logits for final_fcnn_weights
  final_fcnn = tf.matmul(flattened_second_conv, final_fcnn_weights) + final_fcnn_bias

  # if it is training or testing
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
