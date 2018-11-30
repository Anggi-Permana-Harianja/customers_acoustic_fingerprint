'''
- converts a trained checkpoints into a frozen model for mobile inference
- usage: python freeze.py
'''

#import basic modules
import argparse
import os.path
import sys

#import scientific modules
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util

#import files
import input_data
import models

FLAGS = None
	
def create_inference_graph(wanted_words, sample_rate, clip_duration_ms, clip_stride_ms, 
						   window_size_ms, window_stride_ms, feature_bin_count):
	'''
	- creates an audio model with the nodes needed for inference
	- args:
		- wanted_words: comma-separated list of the words we are trying to recognize
		- sample_rate: how many samples per second are in the input data
		- clip_duration_ms: how many samples needs to analyze for the audio pattern
		- clip_stride_ms: how often to run the recognition process. 
		- window_size_ms: time slice duration to estimate frequencies from
		- window_stride_ms: how far apart time slices should be
		- feature_bin_count: number of frequency bands to analyze
	'''

	#get clean word_list
	words_list = input_data.prepare_words_list(wanted_words.split(','))

	#set the model settings
	model_settings = models.prepare_model_settings(len(words_list), sample_rate, clip_duration_ms, 
												   window_size_ms, window_stride_ms, feature_bin_count)

	#create placeholders
	wav_data_placeholder = tf.placeholder(tf.string, [], name = 'wav_data')

	#decode the sample data
	decoded_sample_data = contrib_audio.decode_wav(wav_data_placeholder, 
												   desired_channels = 1, 
												   desired_samples = model_settings['desired_samples'], 
												   name = 'decoded_sample_data')
	#create spectrogram from decoded wav
	spectrogram = contrib_audio.audio_spectrogram(decoded_sample_data.audio, 
												  window_size = model_settings['window_size_samples'], 
												  stride = model_settings['window_stride_samples'], 
												  magnitude_squared = True)

	#produce fingerprint input 
	fingerprint_input = contrib_audio.mfcc(spectrogram, sample_rate, 
										   dct_coefficient_count = model_settings['fingerprint_width'])
	#get fingerprint size
	fingerprint_size = model_settings['fingerprint_size']

	#reshaped fingerprint_size into 2D
	reshaped_input = tf.reshape(fingerprint_input, [-1, fingerprint_size])


	#get logits from our created model in models.py
	#is_training = False since it is for inference and not training
	logits = models.conv_model(reshaped_input, model_settings, is_training = False)

	#create softmaz output to use for inference
	tf.nn.softmax(logits, name = 'labels_softmax')

#main function
def main(_):
	#interactive session
	sess = tf.InteractiveSession()

	#create the frozen model and load its weights
	create_inference_graph(FLAGS.wanted_words, FLAGS.sample_rate, 
						   FLAGS.clip_duration_ms, FLAGS.clip_stride_ms, 
						   FLAGS.window_size_ms, FLAGS.window_stride_ms, 
						   FLAGS.feature_bin_count)
	tf.contrib.quantize.create_eval_graph()

	#load variables
	models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

	#IMPORTANT
	#-------------------------------------------------------------------------
	#turn all the variables into inline constants inside the graph and save it
	frozen_graph_def = graph_util.convert_variables_to_constants(sess, 
																 sess.graph_def, 
																 ['labels_softmax'])

	#write graph
	tf.train.write_graph(frozen_graph_def, 
						 os.path.dirname(FLAGS.output_file), 
						 os.path.basename(FLAGS.output_file),
						 as_text = False)
	#--------------------------------------------------------------------------

	tf.logging.info('waved frozen graph to %s', FLAGS.output_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--sample_rate', 
						type = int, 
						default = 16000, 
						help = 'expected sample rate of the wavs')
	parser.add_argument('--clip_duration_ms', 
						type = int, 
						default = 1000,
						help = 'expected duration in miliseconds of the wavs')
	parser.add_argument('--clip_stride_ms', 
						type = int, 
						default = 30, 
						help = 'how often to run recognition')
	parser.add_argument('--window_size_ms', 
						type = float, 
						default = 30.0, 
						help = 'how long each spectrogram timeslices is')
	parser.add_argument('--window_stride_ms', 
					 	type = float, 
					 	default = 10.0, 
					 	help = 'how long strides is between spectrogram timeslices')
	parser.add_argument('--feature_bin_count', 
						type = int, 
						default = 40, 
						help = 'how many bins of bands for the MFCC fingerprint')
	parser.add_argument('--start_checkpoint', 
						type = str, 
						default = './logs/mfcc.ckpt-800',
						help = 'load specified checkpoint if given')
	parser.add_argument('--wanted_words', 
						type = str, 
						default = 'up, down, left, right, stop', 
						help = 'words to use')
	parser.add_argument('--output_file', 
						type = str, 
						default = './inference_model/inference_graph.pb',
						help = 'where to save frozen graph')
	parser.add_argument('--quantize', 
						type = bool, 
						default = True, 
						help = 'whether to train 8 bit deployment')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)



