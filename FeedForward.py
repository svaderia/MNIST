#!/usr/bin/env python
# @author = 53 68 79 61 6D 61 6C 
# date	  = 07/06/2017

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import timeit
import functools

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class FeedForwardNetwork:

    def __init__(self, layers, learning_rate, batch_size):
        self.layers = layers
        self.lr = learning_rate
        self.batch_size = batch_size

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.layers[0]])
            self.y = tf.placeholder(tf.float32)
    
    def _create_weights(self):
        with tf.name_scope("weights"):
            self.biases = [tf.Variable(tf.random_normal([y])) for y in self.layers[1:]]
            self.weights = [tf.Variable(tf.random_normal([x, y])) for x,y in zip(self.layers[:-1], self.layers[1:])]

    def _feed_forward(self, input_a):
        with tf.name_scope("feed_forward"):
            output_a = input_a
            for w,b in zip(self.weights,self.biases):
                output_a = tf.add(tf.matmul(output_a,w),b)
                if w == self.weights[-1]:
                    break
                output_a = tf.nn.relu(output_a)
            return output_a
    
    def _create_loss(self):
        with tf.name_scope("loss"):
            prediction = self._feed_forward(self.x)
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y)
            self.loss = tf.reduce_mean(entropy)
    
    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _accuracy(self):
        with tf.name_scope("accuracy"):
            prediction = self._feed_forward(self.x)
            correct_label = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_label, 'float'))        

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            self.ep_loss = tf.placeholder(tf.float32)
            tf.summary.scalar("loss", self.ep_loss)
            tf.summary.histogram("histogram loss", self.ep_loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.histogram("histogram accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()
    
    def build_graph(self):
        self._create_placeholders()
        self._create_weights()
        self._create_loss()
        self._create_optimizer()
        self._accuracy()
        self._create_summaries()

def train_model(model, data, epochs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('improved_graph/lr' + str(model.lr), sess.graph)
        n_batch = int(data.train.num_examples/model.batch_size)

        for epoch in range(epochs):
            epoch_loss = 0
            for index in range(n_batch):
                batch_x, batch_y = data.train.next_batch(model.batch_size)
                _ , loss = sess.run([model.optimizer, model.loss], feed_dict = {model.x: batch_x, model.y: batch_y}) 
                epoch_loss += loss
            summary = sess.run(model.summary_op, feed_dict={model.ep_loss: epoch_loss, model.x : data.test.images, model.y: data.test.labels})
            writer.add_summary(summary, epoch )
            print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss)
            print('Epoch' , epoch+1, 'Accuracy:', model.accuracy.eval({model.x: data.test.images, model.y: data.test.labels}))

def main():
    layers = [784, 500, 500, 10]
    epochs = 25
    batch_size = 100
    lr = 0.01
    model = FeedForwardNetwork(layers, lr, batch_size)
    model.build_graph()
    train_model(model, mnist, epochs)
    print("Run `tensorboard --logdir='improved_graph/' to see the summary")

if __name__ == "__main__":
    main()