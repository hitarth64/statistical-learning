import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Function to add bias term to the input and to vectorize the image.
def data_modifier(inp_array):
    shape_vec = inp_array.shape
    inp_array = np.reshape(inp_array, (shape_vec[0],784))
    inp_array = np.c_[np.ones(shape_vec[0]), inp_array]
    inp_array = target_modifier(inp_array)
    return inp_array

# Function to change the data type to float - 32
def target_modifier(inp_array):
    inp_array = inp_array.astype(np.float32)
    return inp_array

with np.load("notMNIST.npz") as data :
	Data, Target = data ["images"], data["labels"]
	posClass = 2
	negClass = 9
	dataIndx = (Target==posClass) + (Target==negClass)
	Data = Data[dataIndx]/255.
	Target = Target[dataIndx].reshape(-1, 1)
	Target[Target==posClass] = 1
	Target[Target==negClass] = 0
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data, Target = Data[randIndx], Target[randIndx]
	trainData, trainTarget = Data[:3500], Target[:3500]
	validData, validTarget = Data[3500:3600], Target[3500:3600]
	testData, testTarget = Data[3600:], Target[3600:]

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

# Hyper-parameters
learning_rate = 0.001
B = 500 # mini_batch_size
reg_param = 0

# Defining placeholders for batch learning
X = tf.placeholder(tf.float32, shape=[None, image_size+1], name='X')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

#theta is the array of weights.
theta = tf.Variable(tf.random_uniform([image_size+1, 1], -1.0, 1.0, seed=42), name="theta")
y_lin = tf.matmul(X, theta, name="y_lin")

reg_error = tf.scalar_mul(reg_param, tf.reduce_sum(tf.square(theta)))
loss = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_lin, labels=y)), reg_error)
y_hat = tf.sigmoid(y_lin, name="predictions")

valid_y_lin = tf.matmul(validData,theta)
valid_acc = tf.reduce_mean(tf.to_float((tf.round(tf.sigmoid(valid_y_lin))[:,0] - validTarget) < 1e-9))
valid_loss = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=valid_y_lin, labels=validTarget)), reg_error)

test_acc = tf.reduce_mean(tf.to_float((tf.round(tf.sigmoid(tf.matmul(testData,theta)))[:,0] - testTarget) < 1e-9))

plt.figure()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

training_losses = []
validation_losses = []
training_accs = []
validation_accs = []
final_test_acc = 0
with tf.Session() as sess:
	sess.run(init)
	sess.run(init_local)

	epoch_ys = np.zeros((3500,1))
	for step in range(0,5000):
		i_batch = (step % 7)*B # Since an epoch consists of 7 batches in our case.
		batch = trainData[i_batch:i_batch+B], trainTarget[i_batch:i_batch+B]
		_,batch_loss,ys = sess.run([training_op,loss,y_hat], feed_dict={X: batch[0] , y: batch[1]})
		epoch_ys[i_batch:i_batch+B,:] = ys

		if (step%7 == 6):
			val_loss,val_acc = sess.run([valid_loss,valid_acc])
			training_losses.append(batch_loss)
			validation_losses.append(val_loss)
			validation_accs.append(val_acc)
			training_acc = np.count_nonzero(np.round(epoch_ys)==trainTarget)/3500
			training_accs.append(training_acc)

			rand_idxs = np.arange(len(trainData))
			np.random.shuffle(rand_idxs)
			trainData = trainData[rand_idxs]
			trainTarget = trainTarget[rand_idxs]
		if (step % 100 == 0 and step > 0):
			print("%d, %f, %f, %f"% (step,training_losses[-1],validation_losses[-1],validation_accs[-1]))

	final_test_acc = sess.run(test_acc)

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

plt.savefig('q2_1_3.png')
plt.show()
