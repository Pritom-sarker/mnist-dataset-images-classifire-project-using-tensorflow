import   tensorflow  as tf
from tensorflow.contrib.layers import fully_connected #Full_connected is use for making nural network..
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import images as img   #This image File convert images to pxils

inputs=28*28  #Each Pic is about 28*28 pxils
h1=100
h2=150
h3=100
out=10        #1-9 total 10 element

X=tf.placeholder(tf.float32,shape=(None,inputs),name='X') # X is none* 786
Y=tf.placeholder(tf.int64,shape=None,name='Y')


keep_prob = 0.5     #This is the  limit rate of dropout



#This peramiter use for normalization
bn_params = {

    'decay': 0.99,
    'updates_collections': None
}


#Nural Net starts here

hidden1=fully_connected(X,h1,scope='Hideen1',activation_fn=tf.nn.elu,normalizer_fn=batch_norm, normalizer_params=bn_params)       # By defult it uses " Relu " As activation function.. we can change activation function

hidden1_drop = dropout(hidden1, keep_prob) #we add dropout after evey layer

hidden2=fully_connected(hidden1,h2,scope='Hideen2',activation_fn=tf.nn.elu,normalizer_fn=batch_norm, normalizer_params=bn_params)

hidden2_drop = dropout(hidden2, keep_prob)

hidden3=fully_connected(hidden2,h3,scope='Hideen3',activation_fn=tf.nn.elu,normalizer_fn=batch_norm, normalizer_params=bn_params)

hidden3_drop = dropout(hidden3, keep_prob)

finalLayer=tf.identity(fully_connected(hidden3,out,scope='Hideen4',activation_fn=None,normalizer_fn=batch_norm, normalizer_params=bn_params), name='Optimizer')
#we dont need any activation function in Final layer





with tf.name_scope('Loss'):
    xenrpy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels= Y,logits=finalLayer) # calculate loss using cross entropy
    loss=tf.reduce_mean(xenrpy)


learning_rate=0.001

with tf.name_scope("Train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate) #gradient decent optimizer
    optmize=optimizer.minimize(loss)

with tf.name_scope('eval'):   # calculate accuracy
    correct=tf.nn.in_top_k( finalLayer,Y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))




#****************** Exucution phase
mnist=input_data.read_data_sets("/tmp/data/") #Pull Mnst data set
epoch=50
step=100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #Initilize global variable
    saver = tf.train.Saver()
    for j in range(1,epoch):

        batch_size = 100
        for i in range(mnist.train.num_examples // batch_size):
            x, y = mnist.train.next_batch(batch_size)    #each time 100 element on X and Y
            sess.run(optmize, feed_dict={X: x, Y: y})


        acc_train = accuracy.eval(feed_dict={X: x, Y: y})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        print(j,"Training acc:: ",acc_train*100," Testing Acc:: ",acc_test*100 )

     #test image
    im = sess.run(finalLayer, feed_dict={X: img.x})
    print(np.argmax(im))

