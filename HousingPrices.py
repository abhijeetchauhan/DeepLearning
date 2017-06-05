
# coding: utf-8

# In[20]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[21]:

df = pd.read_csv("dataHousingPrices.csv")


# In[22]:

df = df.drop(['index','price','sq_price'],axis=1)


# In[23]:

dataframe = df[:10]


# In[24]:

dataframe.loc[:, ("y1")] = [1, 0, 0, 0, 1, 1, 0, 0, 0, 1] # This is our friend's list of which houses she liked
                                                          # 1 = good, 0 = bad
dataframe.loc[:, ("y2")] = dataframe["y1"] == 0           # y2 is the negation of y1
dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)    # Turn TRUE/FALSE values into 1/0
# y2 means we don't like a house
# (Yes, it's redundant. But learning to do it this way opens the door to Multiclass classification)
dataframe # How is our dataframe looking now?


# In[25]:

#prepare data for tensorflow
# tensors are generic version of vectors(1d tensor) and matrices(2d tensor)
# list of list of list of number is 3d tensor
inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
inputY = dataframe.loc[:, ["y1", "y2"]].as_matrix()


# In[26]:

# Parameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = inputY.size


# In[27]:

# Create our computation graph or neural network
# for feature input tensorflow, none means any number of example and 2 is the number of inputs
# placeholder are the gateways for data into our computation graph
x = tf.placeholder(tf.float32,[None,2])

# create weights
# 2x2 float matrix that will be updated through training process
# variable in tf holds and updates parameters
# they are in memory buffers containing tensors
W = tf.Variable(tf.zeros([2,2]))

#Add biases
b = tf.Variable(tf.zeros([2]))   

#multiply weights by inputs and add biases
y_values = tf.add(tf.matmul(x, W), b) 

# apply softmax to value we just create
# softmax is an activation function
# softmax normalizes our values

y = tf.nn.softmax(y_values)

# feed it to output of labels
y_ = tf.placeholder(tf.float32, [None,2])


# In[28]:

# Cost function: Mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[29]:

# Initialize variabls and tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[30]:

# training loop
for i in range(training_epochs):
    sess.run(optimizer,feed_dict={x: inputX, y_:inputY})
    # Display logs per epoch step
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc) #, \"W=", sess.run(W), "b=", sess.run(b)
print "Optimization Finished!"
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'


# In[31]:

sess.run(y, feed_dict={x: inputX })


# In[32]:

sess.run(tf.nn.softmax([1., 2.]))


# In[ ]:



