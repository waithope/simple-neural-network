import os
import numpy as np
import tensorflow as tf
from utils import load_data
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray


# x1 = tf.constant([1, 2, 3, 4])
# x2 = tf.constant([5, 6, 7, 8])

# result = tf.multiply(x1, x2)

# config = tf.ConfigProto(allow_soft_placement=True)
# sess = tf.Session(config=config)
# output = sess.run(result)
# print(output)
root_path = "D:\\VscodeProject\\Python\\simple_nn_tensorflow\\dataset"
train_data_dir = os.path.join(root_path, 'Training')
test_data_dir = os.path.join(root_path, 'Testing')

images, labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)
images, labels = np.array(images), np.array(labels)
test_images, test_labels = np.array(test_images), np.array(test_labels)
print('train dataset', images.shape, labels.shape)
print('test dataset', test_images.shape, test_labels.shape)
# print(images.ndim, images.size)
# print(labels.ndim, labels.size)

# traffic_signs = [300, 2250, 3650, 4000]

# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)

# plt.show()

images28 = np.array([transform.resize(image, (28, 28)) for image in images])
test_images28 = np.array([transform.resize(image, (28, 28)) for image in test_images])

# Convert rgb to gray
images28 = rgb2gray(images28)
test_images28 = rgb2gray(test_images28)

traffic_signs = [300, 2250, 3650, 4000]

# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images28[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)

with tf.Graph().as_default():
    config = tf.ConfigProto(allow_soft_placement=True)
    tf.set_random_seed(42)
    sess = tf.Session(config=config)
    with sess.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
        y = tf.placeholder(dtype=tf.int32, shape=[None])

        images_flat = tf.contrib.layers.flatten(x)

        # Fully connected layer
        with tf.variable_scope('hidden1'):
            h1 = tf.layers.dense(images_flat, 300, tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        with tf.variable_scope('hidden2'):
            h2 = tf.layers.dense(h1, 300, tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        with tf.variable_scope('output'):
            logits = tf.layers.dense(h2, 62, tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        # Define loss
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits))

        # Define accuracy
        with tf.variable_scope('accuracy'):
            predictions = tf.argmax(logits, 1)
            correct_pred = tf.equal(tf.cast(predictions, tf.int32), y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Define training procedure
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # training
        sess.run(tf.global_variables_initializer())

        for i in range(201):
            print('Epochs ', i+1)
            _, loss_, acc = sess.run([train_op, loss, accuracy],
                                    feed_dict={x: images28, y: labels})
            if i % 5 == 0:
                print('loss: ', loss_)
                print('accuracy: ', acc)
        print('training done!!!')

        # evaluating
        pred, acc = sess.run([predictions, accuracy],
                             feed_dict={x: test_images28, y: test_labels})
        print('Evaluation Accuracy: ', acc)





