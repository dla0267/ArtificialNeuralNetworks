import tensorflow as tf
import pdb
import random

#conv_down corresponds to left side of unet
#conv_up corresponds to right side
#run_unet determines which shape parameters to use whether training or testing
#output_layer does final transpose of convolution from last unet layer

def conv_down(input, concat_stack, num_filters, is_training):
        if(num_filters == 64):
            output= tf.nn.conv2d(input, tf.get_variable('firstdown%d' % num_filters, [3,3,1,num_filters], dtype=tf.float32), strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            output= tf.nn.relu(tf.layers.batch_normalization(output, training= is_training))
            output= tf.nn.conv2d(output, tf.get_variable('2ndown%d' % num_filters, [3,3,num_filters,num_filters], dtype=tf.float32), strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        else:
            output= tf.nn.conv2d(input, tf.get_variable('down%d' % num_filters, [3,3,num_filters/2,num_filters], dtype=tf.float32), strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            output= tf.nn.relu(tf.layers.batch_normalization(output, training= is_training))
            output= tf.nn.conv2d(output, tf.get_variable('2nd%d' % num_filters, [3,3,num_filters,num_filters], dtype=tf.float32), strides=[1,1,1,1], padding='SAME', data_format='NHWC')
        output= tf.nn.relu(tf.layers.batch_normalization(output, training= is_training))
        concat_stack.append(output)
        output= tf.layers.max_pooling2d(output, 2, strides=2, padding='SAME')
        return output, concat_stack

def conv_up(input, concat_stack, batch_num, out_dims, num_filters, idx, is_training):
        num_filters= int(num_filters)
        #if(num_filters == 512):   #for more layers
        if(num_filters == 256):
            filter= tf.get_variable('up%d%d' % (idx, out_dims), [2,2,num_filters,num_filters], dtype=tf.float32)   
        else: 
            filter= tf.get_variable('up%d%d' % (idx, out_dims), [2,2,num_filters,num_filters*2], dtype=tf.float32)   
#        pdb.set_trace()
        output= tf.nn.conv2d_transpose(input, filter, tf.stack([batch_num, out_dims*2, out_dims*2, num_filters]), strides=[1,2,2,1], data_format='NHWC')
        stack_img= concat_stack[idx]
        output= tf.concat([output, stack_img], 3)     # concat along 3rd channel which is the feature map channel   
        filter= tf.get_variable('up%d' % (out_dims+random.randint(1,3000)), [3,3,output.get_shape().as_list()[3],num_filters], dtype=tf.float32)   #out_dims/2 when we had pooling
        output= tf.nn.conv2d(output, filter, strides=[1,1,1,1], padding='SAME')
        output= tf.nn.relu(tf.layers.batch_normalization(output, training= is_training))
        filter= tf.get_variable('up%d' % (out_dims+random.randint(1,3000)), [3,3,num_filters,num_filters], dtype=tf.float32)
        output= tf.nn.conv2d(output, filter, strides=[1,1,1,1], padding='SAME')
        output= tf.nn.relu(tf.layers.batch_normalization(output, training= is_training))
        return output, concat_stack

def output_layer(input, batch_num, out_dims, num_filters, is_training):
        filter= tf.get_variable('final%d' % (random.randint(1,20)), [1,1,num_filters,1], dtype=tf.float32)  
        #output= tf.nn.conv2d_transpose(input, filter, tf.stack([batch_num, out_dims, out_dims, 1]), strides=[1,1,1,1], padding='SAME') 
        output= tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')
        output= tf.nn.relu(tf.layers.batch_normalization(output, training= is_training))
        return output

def run_unet(input, concat_stack, num_filters, is_training):
#        output, concat_stack= tf.cond(is_training, lambda: conv_up(input, concat_stack, 128, 64, num_filters, 3, is_training),  #4th param was 12, 3rd was 64 for VALID padding
#                                                   lambda: conv_up(input, concat_stack, 1, 32, num_filters, 3, is_training))   #was 64 instrawd of 512
                                                   
        #num_filters= num_filters/2
        output, concat_stack= tf.cond(is_training, lambda: conv_up(input, concat_stack, 128, 8, num_filters, 2, is_training),  #28
                                                   lambda: conv_up(input, concat_stack, 1, 64, num_filters, 2, is_training))
        num_filters= num_filters/2
        output, concat_stack= tf.cond(is_training, lambda: conv_up(output, concat_stack, 128, 16, num_filters, 1, is_training),  #61
                                                   lambda: conv_up(output, concat_stack, 1, 128, num_filters, 1, is_training))
        num_filters= num_filters/2
        output, concat_stack= tf.cond(is_training, lambda: conv_up(output, concat_stack, 128, 32, num_filters, 0, is_training),  #126
                                                   lambda: conv_up(output, concat_stack, 1, 256, num_filters, 0, is_training))
#        del concat_stack[0:3]       
        del concat_stack[0:2]
        output= tf.cond(is_training, lambda: output_layer(output, 128, 64, num_filters, is_training),
                                     lambda: output_layer(output, 1, 512, num_filters, is_training))
        return output