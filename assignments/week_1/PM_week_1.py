#! /usr/bin/env python

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1) Load the data
def load_space_csv_data(file_name): # Panda function defined for easy data reading
    df = pd.read_csv(file_name, delim_whitespace=True)
    cols = list(df.columns.values)
    return df, cols

df, cols = load_space_csv_data('poverty.txt')
#print cols
#print df.head()

# Pull out the required data sets as arrays
xx1 = df['PovPct'].values
bb = df['Brth15to17'].values

#xx2 = np.column_stack((df['PovPct'].values, df['ViolCrime'].values)) #df['PovPct','ViolCrime'].values
#print xx2.shape

def train(xx,bb):
	# Parameters
	learning_rate = 0.0001
	training_epochs = 1000
	display_step = 50

	# Training Data
	train_X=np.asarray(xx)
	train_Y=np.asarray(bb)
	n_samples = train_X.shape[0]


	# tf Graph Input
	X = tf.placeholder("float")
	Y = tf.placeholder("float")

	# Set model weights
	W = tf.Variable(np.random.randn(), name="weight")
	b = tf.Variable(1.0, name="bias") # Must be altered to ones

	# Construct a linear model
	pred = tf.add(tf.multiply(X, W), b)


	# Mean squared error
	cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
	# Gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		# Fit all training data
		for epoch in range(training_epochs):
			for (x, y) in zip(train_X, train_Y):
				sess.run(optimizer, feed_dict={X: x, Y: y})

			# Display logs per epoch step
			if (epoch+1) % display_step == 0:
				c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
					"W=", sess.run(W), "b=", sess.run(b))

		print("Optimization Finished!")
		training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
		print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

		# Graphic display
		plt.plot(train_X, train_Y, 'ro', label='Original data')
		plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
		plt.xlabel("Pov Pct")
		plt.ylabel("Brth 15to17")
		plt.title("Regression")
		plt.legend()
		plt.show()


train(xx1,bb);
#train(xx2,bb);		