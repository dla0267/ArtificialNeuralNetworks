#import argparse
#from glob import glob

import tensorflow as tf
import pdb
import numpy as np

def input_fn(filename, num_epochs, batch_size):

      def decode_csv(line): 
         defaults= [[0]]*786
         COLUMNS= ['pixel%d' % d for d in range(1,785)]
         #pdb.set_trace()
         record=tf.decode_csv(line, record_defaults=defaults)
         label= record[1]
         del record[0]
         del record[1]
         features= dict(zip(COLUMNS, record))
         return (features, label)
      
      record_defaults= [tf.int32]*785
      #dataset= tf.data.experimental.CsvDataset(filename, record_defaults, header=True, select_cols=range(1,785)).map(decode_csv)
      dataset= tf.data.TextLineDataset(filename).skip(1).map(decode_csv)
      return dataset.shuffle(buffer_size=80000).batch(batch_size).repeat(num_epochs) #.make_one_shot_iterator().get_next()
      
def train_input_fn(filename):
      return input_fn(filename, 20, 100)

def eval_input_fn(filename):
      return input_fn(filename, 1, 100)

def model_fn(features, labels, mode, params):
      input= params['features']
      net= tf.feature_column.input_layer(features, input)
      net= tf.reshape(net, [-1,28,28,1])
      #pdb.set_trace()
      conv1= tf.layers.conv2d(net, 32, 5, padding='same', activation=tf.nn.relu)
      pool1= tf.layers.max_pooling2d(conv1, [2,2], [2,2])
      conv2= tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)
      pool2= tf.layers.max_pooling2d(conv2, [2,2], [2,2])
      pool2_flat= tf.reshape(pool2, [-1, 7*7*64])
      dense= tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
      dropout= tf.layers.dropout(dense, rate=0.4)
      logits= tf.layers.dense(dropout, params['n_classes'])
      predicted= tf.argmax(logits, 1)
      predictions= {'class_ids': predicted[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits}
      loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
      train_op= tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, global_step=tf.train.get_global_step())
      accuracy= tf.metrics.accuracy(labels=labels, predictions=predicted)
      metrics= {'accuracy': accuracy}
      tf.summary.scalar('accuracy', accuracy[1])
      print('did something')
      return tf.estimator.EstimatorSpec(mode, predictions=predictions, train_op=train_op, loss=loss, eval_metric_ops=metrics)

def main(_):
   #pdb.set_trace()
   filename= "C:/Users/seane/Desktop/school/485/finalproject/train.csv"
   #dataset= input_fn(filename, 20, 128)
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
   with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

      #dataset= train_input_fn(filename)
      #pdb.set_trace()
      COLUMNS= ['pixel%d' % d for d in range(1,785)]
      feature_columns= [tf.feature_column.numeric_column(name) for name in COLUMNS]
      #pdb.set_trace()
      model= tf.estimator.Estimator(model_fn, model_dir="/data/coults3/485/model", params={'features': feature_columns, 'hidden_units': [10,10], 'n_classes':10})
      model.train(input_fn=lambda:train_input_fn(filename))
      result= model.evaluate(input_fn=lambda:eval_input_fn(filename))
      #predict= {}
      predictions= model.predict(input_fn=lambda:eval_input_fn(predict_x, 128))
#      for pred_dict, expected in zip(predictions, expected):
#         class= pred_dict['class_ids'][0]
#         probability= pred_dict['probabilites'][class_id]      

if __name__ == '__main__':
    tf.app.run()
