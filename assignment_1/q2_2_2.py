import numpy as np
import tensorflow as tf
import sys
import math
from matplotlib import pyplot as plt

# Function to add bias term to the input and to vectorize the image.
def data_modifier(inp_array):
    shape_vec = inp_array.shape
    inp_array = np.c_[np.ones(shape_vec[0]), inp_array]
    inp_array = inp_array.astype(np.float32)
    return inp_array

def target_modifier(inp_array):
    inp_array = inp_array.astype(np.float32)
    return inp_array

def target_onehot(target):
	out = np.zeros((target.shape[0], 6))
	for i in range(target.shape[0]):
		out[i,np.int(target[i])] = 1
	return out

def data_segmentation(data_path, target_path, task):
	# task = 0 >> select the name ID targets for face recognition task
	# task = 1 >> select the gender ID targets for gender recognition task
	data = np.load(data_path)/255
	data = np.reshape(data, [-1, 32*32])
	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))
	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
			data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
			data[rnd_idx[trBatch + validBatch+1:-1],:]
	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
			target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
			target[rnd_idx[trBatch + validBatch + 1:-1], task]
	return trainData, validData, testData, trainTarget, validTarget, testTarget

classes = ['Lorraine Bracco', 'Gerard Butler', 'Peri Gilpin', 'Angie Harmon', 'Daniel Radcliffe', 'Michael Vartan']

trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data_facescrub.npy', 'target_facescrub.npy', 0)

# Basic parameters
shape_vec = trainData.shape
image_size = shape_vec[1]

# Casting inputs to proper forms
trainData = data_modifier(trainData)
trainTarget = target_modifier(trainTarget)
testData = data_modifier(testData)
testTarget = target_modifier(testTarget)
validData = data_modifier(validData)
validTarget = target_modifier(validTarget)

# Defining placeholders for batch learning
X = tf.placeholder(tf.float32, shape=[None, image_size+1], name='X')
y = tf.placeholder(tf.float32, shape=[None, 6], name='y')

#theta is the array of weights.
theta = tf.Variable(tf.random_uniform([image_size+1, 6], -1.0, 1.0, seed=42), name="theta")
y_lin = tf.matmul(X, theta, name="y_lin")
y_hat = tf.nn.softmax(y_lin, name="predictions")

valid_y_lin = tf.matmul(validData,theta)
valid_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(valid_y_lin),axis=1), validTarget)))
test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(tf.matmul(testData,theta)),axis=1), testTarget)))
train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(tf.matmul(trainData,theta)),axis=1), trainTarget)))

# Hyper-parameters
learning_rate = 0.01
B = 300 # mini_batch_size
reg_param = 0.001

reg_error = tf.scalar_mul(reg_param, tf.reduce_sum(tf.square(theta)))
loss = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_lin, labels=y)), reg_error)

valid_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_y_lin, labels=tf.one_hot(validTarget,6))
valid_loss = tf.add(tf.reduce_mean(valid_entropies), reg_error)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

epoch_len = trainData.shape[0]
n_batches = math.ceil(epoch_len/B)
training_losses = []
validation_losses = []
training_accs = []
validation_accs = []
final_test_acc = 0
init = tf.global_variables_initializer()
failed_image = np.zeros((1, 1025))
net_class = 0
target_class = 0
with tf.Session() as sess:
	sess.run(init)

	epoch_ys = np.zeros((epoch_len,6))
	for step in range(0,5000):
		i_batch = (step % n_batches)*B # Since an epoch consists of 30 batches in our case.
		batch_end = min(i_batch+B, epoch_len)
		batch = trainData[i_batch:batch_end], trainTarget[i_batch:batch_end]
		_,batch_loss,ys = sess.run([training_op,loss,y_hat], feed_dict={X: batch[0] , y: target_onehot(batch[1])})
		epoch_ys[i_batch:i_batch+B,:] = ys

		if ((step+1)% n_batches == 0):
			val_loss,val_acc = sess.run([valid_loss,valid_acc])
			training_losses.append(batch_loss)
			validation_losses.append(val_loss)
			validation_accs.append(val_acc)
			training_acc = sess.run(train_acc)
			training_acc = np.mean(np.equal(np.argmax(epoch_ys,axis=1), trainTarget).astype(float))
			training_accs.append(training_acc)

			rand_idxs = np.arange(len(trainData))
			np.random.shuffle(rand_idxs)
			trainData = trainData[rand_idxs]
			trainTarget = trainTarget[rand_idxs]

	final_test_acc = sess.run(test_acc)

	# find an image the network fails on
	for i in range(0,testData.shape[0]):
		failed_image[0,:] = testData[i,:]
		net_class = np.argmax(sess.run(y_hat, feed_dict={X:failed_image}))
		if net_class != testTarget[i]:
			target_class = testTarget[i]
			break


plt.figure(figsize=(10,5))

plt.subplot(121)
plt.xlabel('Number of Epochs')
plt.ylabel('Cross-entropy loss')
plt.plot(training_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.legend()

plt.subplot(122)
plt.xlabel('Number of Epochs')
plt.ylabel('Classification accuracy')
plt.plot(training_accs, label='Training accuracy')
plt.plot(validation_accs, label='Validation accuracy')
plt.ylim([0,1])
plt.legend()

print("Lowest training loss:", min(training_losses))
print("Lowest validation loss:", min(validation_losses))
print("Highest training accuracy:", max(training_accs))
print("Highest validation accuracy:", max(validation_accs))
print("Test accuracy:", final_test_acc)

plt.tight_layout()

plt.savefig('q2_2_2.png')

plt.figure()

plt.imshow(np.reshape(failed_image[0,1:], (32,32)), cmap='gray')
plt.xlabel("Network output: %d (%s)\nCorrect output: %d (%s)" %
		(net_class, classes[np.int(net_class)], target_class, classes[np.int(target_class)]))

plt.tight_layout()

plt.savefig('q2_2_2_failure.png')

plt.show()
