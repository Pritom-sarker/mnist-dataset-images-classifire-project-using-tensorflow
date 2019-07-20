import  tensorflow as tf
import numpy  as np
from tensorflow.examples.tutorials.mnist import  input_data


#Load The dataset
data=input_data.read_data_sets("temp/data/",one_hot="true")

#size of Hidden layer
h1=50
h2=70
h3=90
out=10

# every time it will fetch 100 element
batch_size=100

#input and out data

X=tf.placeholder(tf.float32,shape=(None,784) ,name='X')
Y=tf.placeholder(tf.int64,shape=None,name='Y')

# initilizr all training variable
hidden_1 = {'w': tf.Variable(tf.random_normal([784, h1])), 'b': tf.Variable(tf.random_normal([h1]))}

hidden_2 = {'w': tf.Variable(tf.random_normal([h1, h2])), 'b': tf.Variable(tf.random_normal([h2]))}

hidden_3 = {'w': tf.Variable(tf.random_normal([h2, h3])), 'b': tf.Variable(tf.random_normal([h3]))}

out_layer = {'w': tf.Variable(tf.random_normal([h3, out])), 'b': tf.Variable(tf.random_normal([out]))}

# part use for nural net

layer_1 = tf.add(tf.matmul(X, hidden_1['w']), hidden_1['b']) # layer_1= x*w + b
layer_1 = tf.nn.relu(layer_1)                               # we are using relu as a activation function

layer_2 = tf.add(tf.matmul(layer_1, hidden_2['w']) , hidden_2['b'])
layer_2 = tf.nn.relu(layer_2)

layer_3 = tf.add(tf.matmul(layer_2, hidden_3['w']), hidden_3['b'])
layer_3 = tf.nn.relu(layer_3)

layer_out = tf.add(tf.matmul(layer_3, out_layer['w']), out_layer['b'], name="Optimizer" )




#calculate loss and Optimize loss

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_out, labels=Y))  #logites = Predict value and lebel= real value
optimizer=tf.train.AdamOptimizer().minimize(loss)



if __name__ == '__main__':
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step=0

            total=50 #Num of loop

            for step in range(total):
                cal_loss=0
                for _ in range(int(data.train.num_examples/batch_size)):
                    x,y=data.train.next_batch(batch_size)
                    _,c=sess.run([optimizer,loss],feed_dict={X:x,Y:y})
                    cal_loss +=c

                cal_acc = tf.equal(tf.argmax(layer_out, 1), tf.argmax(Y, 1))
                acc = tf.reduce_mean(tf.cast(cal_acc, 'float'))
                all=acc.eval({X:data.test.images,Y:data.test.labels})

                print('Step Number:', step, 'loss:', cal_loss,"acc",all)
                saver = tf.train.Saver()
                saver.save(sess, "G:\PycharmProjects\class\mnst\saveMnst50_50.ckpt")



