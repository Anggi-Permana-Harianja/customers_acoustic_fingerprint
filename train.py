'''
prepare input data, models then train
'''
# import modules
import argparse
import os.path
import sys

# import spesific modules
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import gfile #for file writing method

#import files
import input_data
import models

FLAGS = None

def main(_):
	# logging
	tf.logging.set_verbosity(tf.logging.INFO)
	#start tensorflow session
	sess = tf.InteractiveSession()

	# prepare models
	model_settings = models.prepare_model_settings(
						len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
						FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms, 
						FLAGS.window_stride_ms, FLAGS.feature_bin_count)
	# prepare input data
	audio_processor = input_data.AudioProcessor(
						FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage, 
						FLAGS.unknown_percentage, FLAGS.wanted_words.split(','), 
						FLAGS.validation_percentage, FLAGS.testing_percentage, 
						model_settings, FLAGS.summaries_dir)

	fingerprint_size = model_settings['fingerprint_size']

	label_count = model_settings['label_count']

	time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

	# training steps and learning rate hyperparameters
	training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
	learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
	if len(training_steps_list) != len(learning_rates_list):
		raise Exception('len(training_steps_list) should equal len(learning_rates_list)')


	#-------------------------------------------
	# input placeholders to get fingerprint input
	input_placeholder = tf.placeholder(tf.float32, [None, fingerprint_size], name = 'fingerprint_input')
	fingerprint_min, fingerprint_max = input_data.features_range()
	fingerprint_input = tf.fake_quant_with_min_max_args(input_placeholder, 
														fingerprint_min, 
														fingerprint_max)
	#-------------------------------------------

	#-------------------------------------------
	# get logits and drop_out that we alerady code models.py
	logits, dropout_prob = models.conv_model(
							fingerprint_input, 
							model_settings, 
							is_training = True)
	#-------------------------------------------

	#-------------------------------------------
	# define labels for  loss and optimizer
	ground_truth_input = tf.placeholder(tf.int64, [None], name = 'ground_truth_input')
	#-------------------------------------------

	#control dependencies (such as NaN values) while training
	control_dependencies = []
	if FLAGS.check_nans:
		checks = tf.add_check_numerics_ops()
		control_dependencies = [checks]


	#-------------------------------------------
	# BACKPROPAGATION AND TRAINING

	with tf.name_scope('cross_entropy'):
		cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels = ground_truth_input, 
																	logits = logits)
	
	# quantize, create training graph
	tf.contrib.quantize.create_training_graph(quant_delay = 0)

	# train scope with train dependencies
	with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
		# learning rate input placeholder
		learning_rate_input = tf.placeholder(tf.float32, [], 
											 name = 'learning_rate_input')
		# GRADIENT DESCENT OPTIMIZER
		train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)

		# PREDICTED 
		predicted_indices = tf.argmax(logits, 1)

		# ACTUAL/CORRECT PREDICTION
		correct_prediction = tf.equal(predicted_indices, ground_truth_input)

		# CONFUSION MATRIX
		confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices, num_classes = label_count)
	#----------------------------------------------

	#----------------------------------------------
	# EVALUATION
	evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	with tf.get_default_graph().name_scope('eval'):
		#write to graph for summaries evaluation
		tf.summary.scalar('cross_entropy', cross_entropy_mean)
		tf.summary.scalar('accuracy', evaluation_step)
	#----------------------------------------------

	#----------------------------------------------
	# SUMMARIES, GRAPHS
	global_step = tf.train.get_or_create_global_step()
	increment_global_step = tf.assign(global_step, global_step + 1)

	# SAVING checkpoints
	saver = tf.train.Saver(tf.global_variables())

	# WRITE to graphs, change summaries_dir location later
	merged_summaries = tf.summary.merge_all(scope = 'eval')
	train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
	validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

	tf.global_variables_initializer().run()

	#LOAD checkpoints if we have any
	start_step = 1
	if FLAGS.start_checkpoint:
		models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
		start_step = global_step.eval(session = sess)

	#logging
	tf.logging.info('training from step: {}'.format(start_step))

	#---------------------------------------------
	# SAVE graph .pbtxt that will be used in GCP ML engine
	tf.train.write_graph(sess.graph, FLAGS.train_dir, 'mfcc.pbtxt')
	#---------------------------------------------

	# save list of words
	with gfile.GFile(os.path.join(FLAGS.train_dir, 'mfcc_labels.txt'), 'w') as f:
		f.write('\n'.join(audio_processor.words_list))

	#---------------------------------------------
	# SESS RUN for training
	training_steps_max = np.sum(training_steps_list)
	for training_step in xrange(start_step, training_steps_max + 1):
		training_steps_sum = 0
		for i in range(len(training_steps_list)):
			training_steps_sum += training_steps_list[i]
			if training_step <= training_steps_sum:
				learning_rate_value = learning_rates_list[i]
				break

		#pull audio for training
		train_fingerprints, train_ground_truth = audio_processor.get_data(
													FLAGS.batch_size, 0, model_settings, 
													FLAGS.background_frequency, FLAGS.background_volume, 
													time_shift_samples, 'training', sess)

		# RUN the graph
		train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
																		[
																			merged_summaries, 
																			evaluation_step, 
																			cross_entropy_mean, 
																			train_step, 
																			increment_global_step
																		],
																		feed_dict = {
																			fingerprint_input: train_fingerprints,
																			ground_truth_input: train_ground_truth, 
																			learning_rate_input: learning_rate_value,
																			dropout_prob: 0.5
																		}
																		)

		# WRITE THOSE SUMMARIES
		train_writer.add_summary(train_summary, training_step)
		tf.logging.info('step: %d\t rate: %f\t accuracy: %f\t cross entropy: %f' %  (training_step, 
																								learning_rate_value, 
																								train_accuracy * 100, 
																								cross_entropy_value))

		
		#SESS.RUN VALIDATION
		is_last_step = (training_step == training_steps_max)
		if(training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
			set_size = audio_processor.set_size('validation')
			total_accuracy = 0
			total_conf_matrix = None

			for i in xrange(0, set_size, FLAGS.batch_size):
				validation_fingerprints, validation_ground_truth = (audio_processor.get_data(FLAGS.batch_size, 
																		i, model_settings, 0.0, 0.0, 0, 'validation', 
																		sess))

				#VALIDATION SUMMARY, ACCURACY
				validation_summary, validation_accuracy, conf_matrix = sess.run(
																			[merged_summaries, evaluation_step, confusion_matrix],
																			feed_dict = {
																				fingerprint_input: validation_fingerprints, 
																				ground_truth_input: validation_ground_truth,
																				dropout_prob: 1.0
																			}
																			)

				#VALIDATION WRITER
				validation_writer.add_summary(validation_summary, training_step)
				batch_size = min(FLAGS.batch_size, set_size - i)
				total_accuracy += (validation_accuracy * batch_size) / set_size

				if total_conf_matrix is None:
					total_conf_matrix = conf_matrix
				else:
					total_conf_matrix += conf_matrix

			#logging
			tf.logging.info('confusion matrix: %s' % (total_conf_matrix))
			tf.logging.info('step: %d\t validation accuracy = %f (N = %d)' % (training_step, total_accuracy * 100, set_size))

		# SAVE THE MODEL PERIODICALLY
		if(training_step % FLAGS.save_step_interval) == 0 or training_step == training_steps_max:
			checkpoint_path = os.path.join(FLAGS.train_dir, 'mfcc.ckpt')
			tf.logging.info('saving %s-%d', checkpoint_path, training_step)
			saver.save(sess, checkpoint_path, global_step = training_step)

	#--------------------------------------------------------

	#--------------------------------------------------------
	# SESS RUN TESTING
	set_size = audio_processor.set_size('testing')
	tf.logging.info('set size %d', set_size)
	total_accuracy = 0
	total_conf_matrix = None

	for i in xrange(0, set_size, FLAGS.batch_size):
		test_fingerprints, test_ground_truth = audio_processor.get_data(
													FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 
													'testing', sess)
		#get TEST ACCURACY
		test_accuracy, conf_matrix = sess.run([evaluation_step, confusion_matrix], 
											  feed_dict = {fingerprint_input: test_fingerprints, 
											  			   ground_truth_input: test_ground_truth, 
											  			   dropout_prob : 1.0})
		batch_size = min(FLAGS.batch_size, set_size - i)
		total_accuracy += (test_accuracy * batch_size) / set_size

		if total_conf_matrix is None:
			total_conf_matrix = conf_matrix
		else:
			total_conf_matrix += conf_matrix

	tf.logging.info('confusion matrix: %s' % (total_conf_matrix))
	tf.logging.info('final test accuracy: %f (N = %d)' % (total_accuracy * 100, set_size))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#all parsers needed for bazel run
	parser.add_argument('--data_url', 
						type = str, 
						default = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', 
						help = 'url of speech dataset')
	parser.add_argument('--data_dir', 
						type = str, 
						default = './speech_dataset', 
						help = 'location of downloaded data')
	parser.add_argument('--background_volume', 
						type = float, 
						default = 0.1, 
						help = 'how loud the noise, the number between 0 and 1')
	parser.add_argument('--background_frequency', 
						type = float, 
						default = 0.8, 
						help = 'how many samples that need to have background noise mixed in')
	parser.add_argument('--silence_percentage', 
						type = float, 
						default = 10.0, 
						help = 'how much data should be silence')
	parser.add_argument('--unknown_percentage', 
						type = float, 
						default = 10.0, 
						help = 'how much data have unknown words')
	parser.add_argument('--time_shift_ms', 
						type = float, 
						default = 100.0, 
						help = 'range to randomly shift the training audio by in time')
	parser.add_argument('--testing_percentage', 
						type = int, 
						default = 10, 
						help = 'what percentage of .wav to use as test set')
	parser.add_argument('--validation_percentage', 
						type = int, 
						default  = 10, 
						help = 'what percentage of .wav should be use as validation set')
	parser.add_argument('--sample_rate', 
						type = int, 
						default = 16000, 
						help = 'expected sample rate of the wavs')
	parser.add_argument('--clip_duration_ms',
						type = int, 
						default = 1000, 
						help = 'expected duration in milliseconds of the wavs')
	parser.add_argument('--window_size_ms', 
						type = float, 
						default = 30.0, 
						help = 'how long each spectrogram timeslice is')
	parser.add_argument('--window_stride_ms', 
						type = float, 
						default = 10.0, 
						help = 'how far to move in time between spectrogram timeslices')
	parser.add_argument('--feature_bin_count', 
						type = int, 
						default = 40, 
						help = 'how many bins to use for the MFCC fingerprints band')
	parser.add_argument('--how_many_training_steps', 
						type = str, 
						default = '15000, 3000', 
						help = 'how many training loops/epochs')
	parser.add_argument('--eval_step_interval', 
						type = int, 
						default = 400, 
						help = 'how often validation results')
	parser.add_argument('--learning_rate', 
						type = str, 
						default = '0.001, 0.0001', 
						help = 'how large a learning arte to use when training')
	parser.add_argument('--batch_size', 
						type = int, 
						default = 100, 
						help = 'how many items to train with at once')
	parser.add_argument('--summaries_dir', 
						type = str, 
						default = './logs', 
						help = 'where to save summary for tensorboard')
	parser.add_argument('--wanted_words', 
						type = str, 
						default = 'up, down, left, right, stop', 
						help = 'words to use')
	parser.add_argument('--train_dir', 
						type = str, 
						default = './logs',
						help = 'directory to write event logs and checkpoints')
	parser.add_argument('--save_step_interval', 
						type = int, 
						default = 100, 
						help = 'save model checkpoints every save steps')
	parser.add_argument('--start_checkpoint', 
						type = str, 
						default = '', 
						help = 'restore the pre-trained model')
	parser.add_argument('--check_nans', 
						type = bool, 
						default = False, 
						help = 'whether to chack NaNs value while training')
	parser.add_argument('--quantize', 
						type = bool, 
						default = True, 
						help = 'whether to train the model in 8 bit deployment')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
