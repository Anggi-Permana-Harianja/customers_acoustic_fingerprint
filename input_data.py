'''
These codes manages all data sample required for this project.
In the future all voice recording gathered form handheld devices
'''
#import basic modules
import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

#import spesific modules
import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

#import more tensorflow modules
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile #for writing file that contains features
from tensorflow.python.util import compat #compability mode for python 2 vs. 3

#hyperparameters for folder naming and etc
max_num_wav_per_class = 2**27 - 1 #~134M
silence_label = '_silence_'
silence_index = 0
unknown_word_label = '_unknown_'
unknown_word_index = 1
background_noise_dir_name = '_background_noise_'
random_seed = 59185 #for random seed in selection

def prepare_words_list(wanted_words):
  return [silence_label, unknown_word_label] + wanted_words

def which_set(filename, validation_percentage, testing_percentage):
  base_name = os.path.basename(filename) #get the tail of the filename
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  #get percentage hashed
  percentage_hash = ((int(hash_name_hashed, 16) % (max_num_wav_per_class)) * (100.0 / max_num_wav_per_class))

  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'

  return result

def load_wav_file(filename):
  #load those files using placeholder to do so
  used_graph = tf.Graph()
  with tf.Session(graph = used_graph) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    #decode wav
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels = 1)

    return sess.run(wav_decoder, feed_dict = {wav_filename_placeholder: filename})

def save_wav_file(filename, wav_data, sample_rate):
  #save those files using placeholder to do so
  used_graph = tf.Graph()
  with tf.Session(used_graph) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1]) #2-dimension data
    #encode wav
    wav_encoder = contrib_audio.encode_wav(wav_data_placeholder, sample_rate_placeholder)
    #save encoded wav
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)

    return sess.run(wav_saver, feed_dict = {wav_filename_placeholder: filename, 
                                            sample_rate_placeholder: sample_rate, 
                                            wav_data_placeholder: np.reshape(wav_data, (-1, 1))}) #any row with 1 col as vector

def features_range():
  features_min, features_max = -247.0, 30

  return features_min, features_max


class AudioProcessor(object):
  
  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage, 
               wanted_words, validation_percentage, testing_percentage, model_settings, summaries_dir):
    #if given directory for saving wav files
    if data_dir:
      self.data_dir = data_dir
      #download the dataset
      self.download_extract(data_url, data_dir)
      self.prepare_data_index(silence_percentage, unknown_percentage, 
                              wanted_words, validation_percentage, testing_percentage)
      #prepare background noise
      self.prepare_background_data()

    #prepare graph for tensorboard
    self.prepare_processing_graph(model_settings, summaries_dir)

  def download_extract(self, data_url, dest_directory):
    if not data_url:
      return

    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)

      #get the filename
      filename = data_url.split('/')[-1] #get the tail name from given directory
      filepath = os.path.join(dest_directory, filename)

      if not os.path.exists(dest_directory):
        def _progress(count, block_size, total_size):
          sys.stdout.write('downloading.. {} {:.3f}'.format(filename, float(count * block_szie) / float(total_size) * 100.0))

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.logging.error('failed to donwload the dataset')

      statinfo = os.stat(filepath)
      tf.logging.info('dowloaded {:.3f}'.format(statinfo.st_size))

    #extract the .tar file
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, silence_percentage, unknown_percentage, 
                         wanted_words, validation_percentage, testing_percentage):
    #make picking and shuffling in random manner
    random.seed(random_seed)
    wanted_words_index = {}
    for index, wanted_words in enumerate(wanted_words):
      wanted_words_index = index + 2

    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}

    all_words = {}
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()

      if word == background_noise_dir_name:
        continue

      all_words[word] = True

      #set in which partition using which_set function
      set_index = which_set(wav_path, validation_percentage, testing_percentage)

      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})

    if not all_words:
      raise Exception('no .wav found at {}'.format(search_path))

    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))

      for _ in range(silence_size):
        self.data_index[set_index].append({'label': silence_label, 'file': silence_wav_path})

      #pick some unknows to add to each partition so it trained between known and unknown words
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_precentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][ : unknown_size])

    #make sure the order is random
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])

    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = unknown_word_index

    self.word_to_index[silence_label] = silence_index


  def prepare_background_data(self):
    self.background_data = []
    background_dir = os.path.join(self.data_dir, background_noise_dir_name)

    if not os.path.exists(background_dir):
      return self.background_data

    used_graph = tf.Graph()
    with tf.session(graph = used_graph) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels = 1)
      search_path = os.path.join(self.data_dir, background_noise_dir_name, '*.wav')

      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(wav_decoder, feed_dict = {wav_filename_placeholder: wav_path}).audio.flatten() #flatten the data
        self.background_data.append(wav_data)

      if not self.background_data:
        raise Exception ('no background file found')

  def prepare_processing_graph(self, model_settings, summaries_dir):
    with tf.get_default_graph().name_scope('data'):
      desired_samples = model_settings['desired_samples']
      self.wav_filename_placeholder_ = tf.placeholder(tf.string, [], name = 'wav_filename')
      wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
      #graph for decoding wav
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels = 1, 
                                             desired_samples = desired_samples)
      #allow samples volume to be adjusted
      self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [], name = 'foreground_volume')
      scaled_foreground = tf.multiply(wav_decoder.audio, self.foreground_volume_placeholder_)

      #shift the sample start position and pad any gaps with zeros
      self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2], name = 'time_shift_padding')
      self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2], name = 'time_shift_offset')

      #pad with zeros
      padded_foreground = tf.pad(scaled_foreground, self.time_shift_padding_placholder_, 
                                 mode = 'CONSTANT')
      sliced_foreground = tf.slice(padded_foreground, self.time_shift_offset_placeholder_, 
                                   [desired_samples, -1])
      #mix in background noise
      self.background_data_placeholder_ = tf.placeholder(tf.float32, [desired_samples, 1], 
                                                         name = 'background_data')
      self.background_volume_placeholder_ = tf.placeholder(tf.float32, [], name = 'background_volume')

      background_mul = tf.multiply(self.background_data_placeholder_, 
                                   self.background_volume_placeholder_)
      background_add = tf.add(background_mul, sliced_foreground)
      background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

      #--------------------------------------------------
      #run the spectrogram and MFCC to get 2D fingerprint
      spectrogram = contrib_audio.audio_spectrogram(background_clamp, 
                                                    window_size = model_settings['window_size_samples'], 
                                                    stride = model_settings['window_stride_samples'], 
                                                    magnitude_squared = True)
      tf.summary.image('spectrogram', tf.expand_dims(spectrogram, -1), max_outputs = 1)
      self.output_ = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate, 
                                        dct_coefficient_count = model_settings['fingerprint_width'])
      tf.summary.image('mfcc', tf.expand_dims(self.output_, -1), max_outputs = 1)
      #---------------------------------------------------

      self.merged_summaries_ = tf.summary.merge_all(scope = 'data')
      if summaries_dir:
        self.summary_writer_ = tf.summary.FileWriter(summaries_dir + '/data', 
                                                     tf.get_default_graph())

  def set_size(self, mode):
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, background_frequency, 
              background_volume_range, time_shift, mode, sess):
    #pick one of the partitions to choose samples from
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))

    #data and labels will be returned populated
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)

    desired_samples = model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')

    pick_deterministically = (mode != 'training')

    for i in xrange(offset, offset + sample_count):
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))

      sample = candidates[sample_index]

      if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
      else:
        time_shift_amount = 0

      if time_shitf_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]

      input_dict = {self.wav_filename_placeholder_: sample['file'], 
                    self.time_shift_padding_placeholder_: time_shift_padding, 
                    self.time_shift_offset_placeholder_: time_shift_offset, }

      #choose background noise to mix in
      if use_background or sample['label'] == silence_label:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]

        if len(background_samples) <= model_settings['desired_samples']:
          raise ValueError('background samples should equal desired samples')

        background_offset = np.random.randint(0, len(background_samples) - model_settings['desired_samples'])
        background_clipped = background_samples[background_offset : (background_offset + desired_samples)]  
        background_reshaped = background_clipped.reshape([desired_samples, 1])

        if sample['label'] == silence_label:
          background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random_uniform(0, background_volume_range)
        else:  
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0

      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume

      #if we want silence, mute out the main sample and leave the background
      if sample['label'] == silence_label:
        input_dict[self.foreground_volume_placeholder_] = 0
      else:
        input_dict[self.foreground_volume_placeholder_] = 1

      #-----------------------------------------------------
      # run the graph to produce the output audio
      summary, data_tensor = sess.run([self.merged_summaries, self.output_], feed_dict = input_dict)
      self.summary_writer.add_summary(summary)

      data[i - offset, : ] = data_tensor.flatten()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
      #-----------------------------------------------------

    return data, labels


  def get_features_wav(self, wav_filename, model_settings, sess):
    desired_samples = model_settings['desired_samples']

    input_dict = {self.wav_filename_placeholder_: wav_filename, 
                  self.time_shift_padding_placeholder_: [[0, 0], [0, 0]], 
                  self.time_shift_offset_placeholder_: [0, 0], 
                  self.background_data_placeholder_: np.zeros([desired_samples, 1]), 
                  self.background_volume_placeholder_: 0, 
                  self.foreground_volume_placeholder_: 1, }

    #run the graph to produce the output
    data_tensor = sess.run([self.output_], feed_dict = input_dict)

    return data_tensor

  def get_unprocessed_data(self, how_many, model_settings, mode):
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many

    desired_samples = model_settings['model_settings']
    words_list = self.words_list
    data = np.zeros((sample_count, desired_samples))
    labels = []

    used_graph = tf.Graph()
    with tf.Session(used_graph) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels = 1, desired_samples = desired_samples)
      foreground_volume_placeholder = tf.placeholder(tf.float32, [])
      scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume_placeholder)

      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))

        sample = candidates[sample_index]
        input_dict = {wav_filename_placeholder: sample['file']}

        if sample['label'] == silence_label:
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1

        data[i, : ] = sess.run(scaled_foreground, feed_dict = input_dict).flatten()
        label_index = self.word_to_index[sample['label']]
        labels.append(words_list[label_index])

    return data, labels