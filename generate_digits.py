import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist=input_data.read_data_sets("MNIST_data")

def create_conv2d(name, inputs, filter_shape, strides, w_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     b_initializer = tf.truncated_normal_initializer(stddev = 0.02)):
    w = tf.get_variable(name+'_w', filter_shape, dtype=tf.float32, initializer = w_initializer)
    b = tf.get_variable(name+'_b', [filter_shape[-1]], dtype=tf.float32, initializer = b_initializer)
    g = tf.nn.conv2d(inputs, w, strides = strides, padding='SAME')
    g = g+b
    g = tf.contrib.layers.batch_norm(g, epsilon = 1e-5)
    g = tf.nn.leaky_relu(g)
    return g

def create_fully_connected(name, inputs, shape, w_initializer=tf.truncated_normal_initializer(stddev=0.02), 
                                                b_initializer = tf.constant_initializer(0)):
    w = tf.get_variable(name+'_w', shape, initializer = w_initializer)
    b = tf.get_variable(name+'_b', [shape[-1]], initializer=b_initializer)
    output = tf.matmul(inputs, w)+b
    return output

def generator(z, reuse=None):
    with tf.variable_scope('gen',reuse=reuse):  
        z_dim = z.shape[1].value
        g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, g_w1) + g_b1
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
        g1 = tf.nn.relu(g1)
    
        # 1st conv layer
        g2 = create_conv2d("conv2", g1, [3,3,1, z_dim/2], strides = [1,2,2,1])
        g2 = tf.image.resize_images(g2, [56, 56])
    
        # 2nd conv layer
        g3 = create_conv2d("conv3", g2, [3, 3, z_dim/2, z_dim/4], strides = [1,2,2,1])
        g3 = tf.image.resize_images(g3, [56, 56])
    
        # 3rd conv layer
        g4 = create_conv2d("conv4", g3, [1, 1, z_dim/4, 1], strides = [1,2,2,1])
        g4 = tf.sigmoid(g4)
    return g4
    
def discriminator(X,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        # 1st conv layer
        d1 = create_conv2d('conv1', X, [5,5,1,32], strides=[1,1,1,1], b_initializer=tf.constant_initializer(0))
        d1 = tf.nn.max_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
        #2nd conv layer
        d2 = create_conv2d('conv2', d1, [5,5, 32, 64], strides=[1,1,1,1], b_initializer=tf.constant_initializer(0))
        d2 = tf.nn.max_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
        # 1st fully connected layer
        d3 = create_fully_connected('fully_connected1', tf.reshape(d2,(-1,7 * 7 * 64 )), [7 * 7 * 64, 1024])
        
        # 2nd fully connected layer
        logits = create_fully_connected("fully_connected2", d3, [1024,1])
        output = tf.sigmoid(logits)
    return output, logits    

tf.reset_default_graph()

real_images=tf.placeholder(tf.float32,shape=[None,28,28,1])
z=tf.placeholder(tf.float32,shape=[None,100])

G=generator(z)
D_output_real,D_logits_real=discriminator(real_images)
D_output_fake,D_logits_fake=discriminator(G,reuse=True)

def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss=loss_func(D_logits_real,tf.ones_like(D_logits_real)*0.9) #Smoothing for generalization
D_fake_loss=loss_func(D_logits_fake,tf.zeros_like(D_logits_real))
D_loss=D_real_loss+D_fake_loss

G_loss= loss_func(D_logits_fake,tf.ones_like(D_logits_fake))

lr=0.001

#Do this when multiple networks interact with each other
tvars=tf.trainable_variables()  #returns all variables created(the two variable scopes) and makes trainable true
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

D_trainer=tf.train.AdamOptimizer(lr).minimize(D_loss,var_list=d_vars)
G_trainer=tf.train.AdamOptimizer(lr).minimize(G_loss,var_list=g_vars)

batch_size=100
epochs=10
init=tf.global_variables_initializer()

samples=[] #generator examples

saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        saver.restore(sess, './model.ckpt')
    except Exception:
        sess.run(init)
    for epoch in range(epochs):
        num_batches=mnist.train.num_examples//batch_size
        for i in range(num_batches):
            batch=mnist.train.next_batch(batch_size)
            batch_images=batch[0].reshape((batch_size,28,28,1))
            batch_z= np.random.normal(size=[batch_size, 100]) 
            _, d_loss=sess.run([D_trainer, D_loss],feed_dict={real_images:batch_images,z:batch_z})            
        sample_z=np.random.normal(size=[1, 100])
        _, g_loss=sess.run([G_trainer, G_loss],feed_dict={z:batch_z})        
        print("on epoch{}, d_loss:{}, g_loss:{}".format(epoch, d_loss, g_loss))
        gen_sample=sess.run(generator(z,reuse=True),feed_dict={z:sample_z})       
        samples.append(gen_sample)
    save_path = saver.save(sess, './model.ckpt')

plt.imshow(samples[0])
plt.imshow(samples[9])

