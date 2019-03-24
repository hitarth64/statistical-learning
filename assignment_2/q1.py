import numpy as np
import tensorflow as tf
import sys
import math
import collections
from matplotlib import pyplot as plt

Dataset = collections.namedtuple('Dataset',
		['X_train', 'y_train', 'X_test', 'y_test', 'X_valid', 'y_valid'])

def reset_graph(seed=42):
	tf.reset_default_graph()
	tf.set_random_seed(seed)
	np.random.seed(seed)

def load_data(digit_range = None):
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
	X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
	y_train = y_train.astype(np.int32)
	y_test = y_test.astype(np.int32)
	X_valid, X_train = X_train[:5000], X_train[5000:]
	y_valid, y_train = y_train[:5000], y_train[5000:]

	if digit_range:
		X_train = X_train[np.isin(y_train, digit_range)]
		y_train = y_train[np.isin(y_train, digit_range)]
		X_valid = X_valid[np.isin(y_valid, digit_range)]
		y_valid = y_valid[np.isin(y_valid, digit_range)]
		X_test = X_test[np.isin(y_test, digit_range)]
		y_test = y_test[np.isin(y_test, digit_range)]

	return Dataset(X_train, y_train, X_valid, y_valid, X_test, y_test)

image_size = 28*28
n_classes = 5

# Planning to add parameters to alter model based on the problem in question
def model(n_hidden=100, activation=tf.nn.elu, bn=False, bn_momentum=0.9, dropout=False, dropout_rate=0.5):
	X = tf.placeholder(tf.float32, shape=[None, image_size], name='X')
	training = tf.placeholder_with_default(False, shape=(), name='training')
	he_init = tf.variance_scaling_initializer()
	def hidden(inputs, name):
		if bn:
			pre_norm = tf.layers.dense(inputs, n_hidden, activation=activation,
					kernel_initializer=he_init, name=name)
			if dropout:
				do_layer = tf.layers.dropout(pre_norm, rate=dropout_rate,
						training=training, name=name+"_do")
				return tf.layers.batch_normalization(do_layer, momentum=bn_momentum,
						training=training, name=name+"_bn")
			return tf.layers.batch_normalization(pre_norm, momentum=bn_momentum,
					training=training, name=name+"_bn")
		if dropout:
			do_layer = tf.layers.dropout(inputs, rate=dropout_rate,
					training=training, name=name+"_do")
			return tf.layers.dense(do_layer, n_hidden, activation=activation,
					kernel_initializer=he_init, name=name)
		return tf.layers.dense(inputs, n_hidden, activation=activation,
				kernel_initializer=he_init, name=name)
	hidden1 = hidden(X, "hidden1")
	hidden2 = hidden(hidden1, "hidden2")
	hidden3 = hidden(hidden2, "hidden3")
	hidden4 = hidden(hidden3, "hidden4")
	hidden5 = hidden(hidden4, "hidden5")
	logits = tf.layers.dense(hidden5, n_classes, kernel_initializer=he_init, name="logits")
	return X, logits, training

def shuffle_batch(xs, ys, batch_size):
	rnd_idx = np.arange(np.shape(xs)[0])
	np.random.shuffle(rnd_idx)
	return ((xs[rnd_idx[batch_size*n:batch_size*(n+1)]],ys[rnd_idx[batch_size*n:batch_size*(n+1)]])
		for n in range(0,math.floor(np.shape(xs)[0]/batch_size)))

def evaluate(X, logits, training, data, n_epochs=1000, batch_size=20, learning_rate=0.005, es_epochs=20, run_name="q1"):
	y = tf.placeholder(tf.int32, shape=[None], name='y')
	classification = tf.to_int32(tf.argmax(tf.nn.softmax(logits),axis=1))
	accuracy = tf.reduce_mean(tf.to_float(tf.equal(classification, y)))
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
	training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	tf.add_to_collection("training_op", training_op)
	extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

	saver = tf.train.Saver()
	init = tf.global_variables_initializer()

	train_losses = []
	train_accs = []
	val_losses = []
	val_accs = []
	failed_image = np.zeros((1, image_size))
	net_class = 0
	target_class = 0
	test_acc = 0

	with tf.Session() as sess:
		def collect_data():
			train_loss,train_acc = sess.run([loss,accuracy], feed_dict={X:data.X_train, y:data.y_train})
			val_loss,val_acc = sess.run([loss,accuracy], feed_dict={X:data.X_valid, y:data.y_valid})
			train_losses.append(train_loss)
			train_accs.append(train_acc)
			val_losses.append(val_loss)
			val_accs.append(val_acc)
			return (train_loss,train_acc,val_loss,val_acc)
		init.run()
		best_val_loss = np.infty
		best_val_acc = np.infty
		best_train_loss = np.infty
		best_train_acc = np.infty
		checks_without_progress = 0
		collect_data()
		for epoch in range(n_epochs):
			for X_batch, y_batch in shuffle_batch(data.X_train, data.y_train, batch_size):
				sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, y: y_batch, training: True})
			train_loss,train_acc,val_loss,val_acc = collect_data()
			print(epoch, val_loss, val_acc)
			if val_loss < best_val_loss:
				save_path = saver.save(sess, "./model.ckpt")
				best_val_loss = val_loss
				best_val_acc = val_acc
				best_train_loss = train_loss
				best_train_acc = train_acc
				checks_without_progress = 0
			else:
				checks_without_progress += 1
				if checks_without_progress > es_epochs:
					print("Early stopping!")
					print(epoch, best_val_loss, best_val_acc)
					break
			if epoch % 100 == 0:
				save_path = saver.save(sess, "./model-%d.ckpt" % epoch)

		# Restore best network
		saver.restore(sess, "./model.ckpt")

		test_acc = sess.run([accuracy], feed_dict={X:data.X_test, y:data.y_test, training: False})[0]
		print("Final training and validation losses:", train_losses[-1], val_losses[-1])
		print("Best network training and validation losses:", best_train_loss, best_val_loss)
		print("Best network training and validation accuracies:", best_train_acc, best_val_acc)
		print("Test accuracy:", test_acc)

		# find an image the network fails on
		for i in range(0,data.X_test.shape[0]):
			failed_image[0,:] = data.X_test[i,:]
			net_class = sess.run(classification, feed_dict={X:failed_image, y:data.y_test[i:i+1]})
			if net_class != data.y_test[i]:
				target_class = data.y_test[i]
				break

	plt.figure(figsize=(10,5))

	plt.subplot(121)
	plt.xlabel('Number of Epochs')
	plt.ylabel('Cross-entropy loss')
	plt.plot(train_losses, 'k-', label='Training loss')
	plt.plot(val_losses, 'k--', label='Validation loss')
	plt.legend()

	plt.subplot(122)
	plt.xlabel('Number of Epochs')
	plt.ylabel('Classification error (%)')
	plt.plot([(1-a)*100 for a in train_accs], 'k-', label='Training classification error')
	plt.plot([(1-a)*100 for a in val_accs], 'k--', label='Validation classification error')
	plt.legend()

	plt.tight_layout()
	plt.savefig(run_name + '.png')

	plt.figure()
	plt.imshow(np.reshape(failed_image, (28,28)), cmap='gray')
	plt.xlabel("Network output: %d (%s)\nCorrect output: %d (%s)" %
			(net_class, net_class, target_class, target_class))
	plt.tight_layout()
	plt.savefig(run_name + '_failure.png')

def usage():
	print("python q1.py [n]")
	print("n is 2, 3, 4, or 5 - for question 1.2.n")

reset_graph()
if len(sys.argv) != 2:
	usage()
elif sys.argv[1] == '2':
	X,logits,training = model()
	evaluate(X, logits, training, load_data(range(1,5)), run_name="q1_2_2")
elif sys.argv[1] == '3':
	X,logits,training = model(n_hidden=50, activation=tf.nn.relu)
	evaluate(X, logits, training, load_data(range(1,5)), batch_size=20, learning_rate=0.005, es_epochs=20, run_name="q1_2_3a")
	reset_graph()
	X,logits,training = model(n_hidden=140, activation=lambda x: tf.nn.leaky_relu(x,alpha=0.1))
	evaluate(X, logits, training, load_data(range(1,5)), batch_size=500, learning_rate=0.1, es_epochs=30, run_name="q1_2_3b")
elif sys.argv[1] == '4':
	for momentum in [0.85,0.9,0.95,0.99]:
		reset_graph()
		X,logits,training = model(bn=True, bn_momentum=momentum)
		evaluate(X, logits, training, load_data(range(1,5)), run_name="q1_2_4_%.2f"%momentum)
elif sys.argv[1] == '5':
	for dropout_rate in [0.1,0.3]:
		reset_graph()
		X,logits,training = model(dropout=True, dropout_rate=dropout_rate)
		evaluate(X, logits, training, load_data(range(1,5)), run_name="q1_2_5_do_%.1f" % dropout_rate)
	for dropout_rate in [0.1,0.3]:
		reset_graph()
		X,logits,training = model(dropout=True, dropout_rate=dropout_rate, bn=True, bn_momentum=0.99)
		evaluate(X, logits, training, load_data(range(1,5)), run_name="q1_2_5_bn_do_%.1f" % dropout_rate)
else:
	usage()
