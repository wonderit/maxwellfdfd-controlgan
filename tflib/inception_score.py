# From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import tflib.fid as fid

MODEL_DIR = './tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None
pool3 = None

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 1
  with tf.compat.v1.Session() as sess:
    preds = []
    features = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):

        # sys.stdout.write(".")
        # sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred, feature = sess.run([softmax, pool3], {'ExpandDims:0': inp})
        features.append(feature)
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    features = np.concatenate(features, 0)
    scores = []
    '''
    features = np.squeeze(np.array(features))
    features_mean = np.mean(features, 0)
    print(features_mean)
    np.save('./cifar10_mean', features_mean)
    features_norm = (features - np.expand_dims(features_mean, 0)).reshape((-1,2048))
    features_cov = np.cov(np.transpose(features_norm))
    np.save('./cifar10_cov', features_cov)
    '''
    features = np.squeeze(np.array(features))
    features_mean_cifar10 = np.load('./cifar10_mean.npy').reshape((2048))
    features_cov_cifar10 = np.squeeze(np.load('./cifar10_cov.npy'))
    features_mean = np.mean(features, 0).reshape((2048))
    diff = features_mean - features_mean_cifar10
    features_norm = (features - np.expand_dims(features_mean, 0)).reshape((-1,2048))
    features_cov = np.cov(np.transpose(features_norm))
    '''
    covmean, _ = linalg.sqrtm(features_cov.dot(features_cov_cifar10), disp=False)
    term1 = diff.dot(np.transpose(diff)).reshape(1)
    term2 = (np.trace(features_cov_cifar10) + np.trace(features_cov) - 2.0*np.trace(covmean)).reshape(1)
    '''
    _fid = fid.frechet_distance(features_mean_cifar10, features_cov_cifar10, features_mean, features_cov)
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores), _fid

# This function is called automatically.
def _init_inception():
  global softmax
  global pool3
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.io.gfile.GFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.graph_util.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.compat.v1.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.shape
            #shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.reshape(pool3, [1,pool3.shape[-1]]), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()
