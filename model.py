import time

from utils import *
from six.moves import xrange
import numpy as np
from conv_utils import *

#input [batch, 28, 28, 1] tensor
#output [batch, 10] tensor
#2 convolutional layers (conv, relu, pool)
#1 dense layer with flattened input
#output vector
#softmax output from dense layer
#cross entropy the softmax
def mnist(input, is_training, output_channels=1):
   conv1= tf.layers.conv2d(input, 32, 5, padding='same', activation=tf.nn.relu)
   pool1= tf.layers.max_pooling2d(conv1, [2,2], [2,2])
   conv2= tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
   pool2= tf.layers.max_pooling2d(conv2, [2,2], [2,2])
   pool2_flat= tf.reshape(pool2, [-1, 7*7*64])
   dense= tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
   dropout= tf.layers.dropout(dense, rate=0.4, training=is_training)
   logits= tf.layers.dense(dropout, units=10)
   return logits

class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=1):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, [None, 28,28, 1])   #input features
        self.Y = mnist(self.X, is_training=self.is_training)       #network classification labels (10)
        self.Y_= tf.placeholder(tf.float32, [None,10])             #ground truth label            (10)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.Y_, self.Y))
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

 
    def train(self, features, labels, sess, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=1):
        # assert data range is between 0 and 1 
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        #img= tf.summary.image('denoised image', self.Y)
        writer = tf.summary.FileWriter('./logs_unet', self.sess.graph)
        merged = tf.summary.merge_all()
        #summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        #self.evaluate(iter_num, ndct_eval_data, ldct_eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
        #              summary_writer=writer, summ_img=img)  # eval_data value range is 0-255
        for epoch in xrange(start_epoch, epoch):
            #pdb.set_trace()
            features= tf.random_shuffle(features, seed=1)
            labels= tf.random_shuffle(labels, seed=1)
            for batch_id in xrange(start_step, 50):
                img = features[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                label= labels[batch_id * batch_size:(batch_id + 1) * batch_size, :]
                img= sess.run(img) / 255
                label= sess.run(label)
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.lr: lr[epoch],
                                                            self.is_training: True,
                                                            self.X: img,
                                                            self.Y_: label})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.15f"
                      % (epoch + 1, batch_id + 1, 50, time.time() - start_time, loss))
                iter_num += 1
                #writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                #self.evaluate(iter_num, ndct_eval_data, ldct_eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                #              summary_writer=writer, summ_img=img)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0