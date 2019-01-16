import argparse
import numpy as np
import struct
from utils import *
from PIL import Image
import glob
import pdb
import tensorflow as tf


parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='f', default='/home/NETID/coults3/train.csv', help='FLT file')
parser.add_argument('--save_dir', dest='save_dir', default='/home/NETID/coults3/', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=28, help='patch size')
parser.add_argument('--size', dest='size', default= 28, help='image size')
parser.add_argument('--stride', dest='stride', type=int, default=1, help='stride')
parser.add_argument('--offset', dest='offset', type=int, default=0,help='offset')
parser.add_argument('--trainset_size', dest='trainset_size', type=int, default=60000, help='batch size')

args= parser.parse_args()

AUG_DATA = 1

def makePatches():
    arr= []
    size= args.size   #all images assumed to be the same size
    numPatches= int(np.square((size - args.pat_size) / args.stride))+1 #######################added +1 for size 512 patches
    #pdb.set_trace()
    inputs = np.zeros((numPatches*args.trainset_size, args.pat_size, args.pat_size, 1), dtype="f")

    #for patch size 128, stride length 8 we have ((512-128)/8)^2=2304 patches per image
    # for training set of 50 images we have 115200 patches overall
    #and batch size of 900 (900 batches) means 128 patches per batch

    count= 0

    #[batch, 28, 28, 1]
    for i in range(1,60000)
        pdb.set_trace()
        filename_queue= tf.train.string_input_producer(["train.csv"])
        reader= tf.TextLineReader()
        key,value= reader.read(filename_queue)
        defaults= [0 for i in range(1,784)]
        train_img=[]
        train_img=tf.decode_csv(value, record_defaults=defaults)
        label=train_img[1]
        features=tf.to_float(tf.stack(train_img[2:]))

        for x in range(0 + args.offset, 28, args.stride):
            for y in range(0 + args.offset, 28, args.stride):
                inputs[count, :, :, :] = data_augmentation(features[(x+1)*y], \
                      0)
                count+= 1

    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, "mnist_pats"), inputs)
    print( "size of inputs tensor = " + str(inputs.shape))
    
    exit(0)

if __name__ == '__main__':
    makePatches()       
