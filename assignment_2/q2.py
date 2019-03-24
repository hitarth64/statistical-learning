import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def data_loader():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]
    X_train2_full = X_train[y_train >= 5]
    y_train2_full = y_train[y_train >= 5] - 5
    X_valid2_full = X_valid[y_valid >= 5]
    y_valid2_full = y_valid[y_valid >= 5] - 5
    X_test2 = X_test[y_test >= 5]
    y_test2 = y_test[y_test >= 5] - 5
    
    return(X_train2_full, y_train2_full, X_valid2_full, y_valid2_full, X_test2, y_test2)

def shuffle_batch(xs, ys, batch_size):
    rnd_idx = np.arange(np.shape(xs)[0])
    np.random.shuffle(rnd_idx)
    return ((xs[rnd_idx[batch_size*n:batch_size*(n+1)]],ys[rnd_idx[batch_size*n:batch_size*(n+1)]])
        for n in range(0,math.floor(np.shape(xs)[0]/batch_size)))

def sample_n_instances_per_class(X, y, n):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y==label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
	
def update_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=20, es_epochs=20, n_epochs=1000, part=2, subpart = 1):
    
    #part has 3 allowed values - '2', '3' and '4' : each corresponding to sub-questions 2.2.2, 2.2.2, 2.2.4!
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_loss = np.infty
    best_loss_acc = np.infty
    checks_without_progress = 0

    with tf.Session() as sess:

        #Restoring previous session and restoring variables values
        saver = tf.train.import_meta_graph("model.ckpt.meta")
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        
        #with graph.as_default():
        saver1 = tf.train.Saver()
        
        # Getting pointers to inputs, outputs and weights 
        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")
        weights = graph.get_tensor_by_name("hidden5/kernel:0")


        if part==2:
            output_tensor = graph.get_tensor_by_name("logits/BiasAdd:0")
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="logits")

        elif part==3:
            hidden4_out = graph.get_tensor_by_name("hidden4/BiasAdd:0")
            hidden4_out
            output_tensor = tf.layers.dense(hidden4_out, 5, name="out_3")
            init_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="out_3")
            sess.run(tf.initialize_variables(init_vars))
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="out_3")

        elif part==4:
            hidden4_out = graph.get_tensor_by_name("hidden4/BiasAdd:0")
            output_tensor = tf.layers.dense(hidden4_out, 5, name="out_4")
            init_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="out_4")
            sess.run(tf.initialize_variables(init_vars))
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="out_4|hidden4|hidden3")

        #for op in tf.get_default_graph().get_operations():
         #   print(op.name)

        #print(weights.eval()) #-> meant to check if we are freezing the layers or not

        # Defining accuracy, classification, loss functions and the optimizer
        def class_loss_opt(logits):
            #logits is the output layer tensor
            classification = tf.to_int32(tf.argmax(tf.nn.softmax(logits),axis=1))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(classification, y)))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
            opt2 = tf.train.AdamOptimizer(learning_rate=0.005, name = "opt2")
            training_op_2 = opt2.minimize(loss,var_list = train_vars, name = "training_op_2")
            # Initializing optimizer's variables since we aren't using the one imported from previous model.
            sess.run(tf.variables_initializer(opt2.variables()))
            return classification, accuracy, loss, training_op_2

        # Establishing operations and optimizers
        classification, accuracy, loss, training_op_2 = class_loss_opt(output_tensor)

        def collect_data(X_train,y_train,X_valid,y_valid,train_losses,train_accs,val_losses,val_accs):
            train_loss,train_acc = sess.run([loss,accuracy], feed_dict={X:X_train, y:y_train})
            val_loss,val_acc = sess.run([loss,accuracy], feed_dict={X:X_valid, y:y_valid})
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            return (train_loss,train_acc,val_loss,val_acc)

        collect_data(X_train,y_train,X_valid,y_valid,train_losses,train_accs,val_losses,val_accs)

        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run([training_op_2], feed_dict={X: X_batch, y: y_batch})
            train_loss,train_acc,val_loss,val_acc = collect_data(X_train,y_train,X_valid,y_valid,train_losses,train_accs,val_losses,val_accs)
            print(epoch, val_loss, val_acc)
            if val_loss < best_loss:
                save_path = saver1.save(sess, "./tf/tl_model.ckpt")
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_loss = val_loss
                best_loss_acc = val_acc
            else:
                checks_without_progress += 1
                if checks_without_progress > es_epochs:
                    print("Early stopping!")
                    print(epoch, best_loss, best_loss_acc)
                    break

        saver1.restore(sess, "./tf/tl_model.ckpt")
        test_acc = sess.run([accuracy], feed_dict={X:X_test, y:y_test})
        print("Final training and validation losses:", train_losses[-1], val_losses[-1])
        print("Best network training and validation losses:", best_train_loss, best_loss)
        print("Best network training and validation accuracies:", best_train_acc, best_loss_acc)
        print("Test accuracy:", test_acc)

        #print(weights.eval()) #-> meant to check if we are freezing the layers or not
    
    plt.figure(figsize=(10,5))

    plt.subplot(121)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Cross-entropy loss')
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()

    plt.subplot(122)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Classification accuracy')
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.ylim([0,1])
    plt.legend()
    # - ignore this comment
    plt.tight_layout()
    if part == 2 and subpart == 1:
        file_name = 'q2_2_'+str(part)+'_full_dataset.png'
    elif part ==2 and subpart == 2:
        file_name = 'q2_2_'+'sampled_dataset.png'
    else:
        file_name = 'q2_2_'+str(part)+'.png'
    plt.savefig(file_name)

    plt.show()
    reset_graph()

X_train2_full, y_train2_full, X_valid2_full, y_valid2_full, X_test, y_test = data_loader()
X_sampled, y_sampled = sample_n_instances_per_class(X_train2_full, y_train2_full, 100)

print("Results for 2.2.2 with full dataset: ") 
update_evaluate(X_train2_full, y_train2_full, X_valid2_full, y_valid2_full, X_test, y_test, part =2)

print("------------------------------------"+'\n')
print("Results for 2.2.2 with sampled dataset: ")
update_evaluate(X_sampled, y_sampled, X_valid2_full, y_valid2_full, X_test, y_test, part =2, subpart = 2)

print("------------------------------------"+'\n')
print("Results for 2.2.3: ")
update_evaluate(X_train2_full, y_train2_full, X_valid2_full, y_valid2_full, X_test, y_test, part =3)

print("------------------------------------"+'\n')
print("Results for 2.2.4: ")
update_evaluate(X_train2_full, y_train2_full, X_valid2_full, y_valid2_full, X_test, y_test, part =4)
