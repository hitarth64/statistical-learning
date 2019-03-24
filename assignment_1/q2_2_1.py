import numpy as np
import tensorflow as tf
import sys
from matplotlib import pyplot as plt

# Function to add bias term to the input and to vectorize the image.
def data_modifier(inp_array):
    shape_vec = inp_array.shape
    inp_array = np.reshape(inp_array, (shape_vec[0],784))
    inp_array = np.c_[np.ones(shape_vec[0]), inp_array]
    inp_array = inp_array.astype(np.float32)
    return inp_array

def target_modifier(inp_array):
    inp_array = inp_array.astype(np.float32)
    return inp_array

def target_onehot(target):
	out = np.zeros((target.shape[0], 10))
	for i in range(target.shape[0]):
		out[i,np.int_(target[i])] = 1
	return out

classes = ['A','B','C','D','E','F','G','H','I','J']

with np.load("notMNIST.npz") as data :
	Data, Target = data ["images"], data["labels"]
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data = Data[randIndx]/255.
	Target = Target[randIndx]
	trainData, trainTarget = Data[:15000], Target[:15000]
	validData, validTarget = Data[15000:16000], Target[15000:16000]
	testData, testTarget = Data[16000:], Target[16000:]

# Basic parameters
shape_vec = trainData.shape
image_size = shape_vec[1] * shape_vec[2]

# Casting inputs to proper forms
trainData = data_modifier(trainData)
trainTarget = target_modifier(trainTarget)
testData = data_modifier(testData)
testTarget = target_modifier(testTarget)
validData = data_modifier(validData)
validTarget = target_modifier(validTarget)

# Defining placeholders for batch learning
X = tf.placeholder(tf.float32, shape=[None, image_size+1], name='X')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

#theta is the array of weights.
theta = tf.Variable(tf.random_uniform([image_size+1, 10], -1.0, 1.0, seed=42), name="theta")
y_lin = tf.matmul(X, theta, name="y_lin")
y_hat = tf.nn.softmax(y_lin, name="predictions")

valid_y_lin = tf.matmul(validData,theta)
valid_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(valid_y_lin),axis=1), validTarget)))
test_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(tf.matmul(testData,theta)),axis=1), testTarget)))
train_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.nn.softmax(tf.matmul(trainData,theta)),axis=1), trainTarget)))

# Hyper-parameters
learning_rate = 0.001
B = 500 # mini_batch_size
reg_param = 0.01

reg_error = tf.scalar_mul(reg_param, tf.reduce_sum(tf.square(theta)))
loss = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_lin, labels=y)), reg_error)

valid_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_y_lin, labels=tf.one_hot(validTarget,10))
valid_loss = tf.add(tf.reduce_mean(valid_entropies), reg_error)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

training_losses = []
validation_losses = []
training_accs = []
validation_accs = []
final_test_acc = 0
failed_image = np.zeros((1, 785))
net_class = 0
target_class = 0
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	epoch_ys = np.zeros((15000,10))
	for step in range(0,5000):
		i_batch = (step % 30)*B # Since an epoch consists of 30 batches in our case.
		batch = trainData[i_batch:i_batch+B], trainTarget[i_batch:i_batch+B]
		_,batch_loss,ys = sess.run([training_op,loss,y_hat], feed_dict={X: batch[0] , y: target_onehot(batch[1])})
		epoch_ys[i_batch:i_batch+B,:] = ys

		if ((step+1)%30 == 0):
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
plt.legend()

plt.tight_layout()

print("Lowest training loss:", min(training_losses))
print("Lowest validation loss:", min(validation_losses))
print("Highest training accuracy:", max(training_accs))
print("Highest validation accuracy:", max(validation_accs))
print("Test accuracy:", final_test_acc)

plt.savefig('q2_2_1.png')

plt.figure()

plt.imshow(np.reshape(failed_image[0,1:], (28,28)), cmap='gray')
plt.xlabel("Network output: %d (%s)\nCorrect output: %d (%s)" %
		(net_class, classes[np.int(net_class)], target_class, classes[np.int(target_class)]))
plt.tight_layout()

plt.savefig('q2_2_1_failure.png')
plt.show()
