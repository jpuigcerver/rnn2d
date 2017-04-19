import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'

import numpy as np
import theano
import theano.tensor as T
import time
from scipy.ndimage import imread
from scipy.misc import imresize
from returnn import Model, Collapse, Loss, SGD
from Util import decode

np.random.seed(1234)

img_list = [ imread('../assets/labour.png').astype(np.float32),
             imread('../assets/tomorrow.png').astype(np.float32) ]
max_h, max_w = 0, 0
for i, img in enumerate(img_list):
    h, w = img.shape
    img_list[i] = imresize(img, [h / 16, w / 16])
    max_h = max(max_h, img_list[i].shape[0])
    max_w = max(max_w, img_list[i].shape[1])

labels = ['a', 'b', 'l', 'm', 'o', 'r', 't', 'u', 'w']
y = np.array(
    [
        3, 1, 2, 5, 8, 6,        # l a b o u r
        7, 5, 4, 5, 6, 6, 5, 9   # t o m o r r o w
    ], dtype=np.int32)
ylen = np.array([6, 8], dtype=np.int32)

z = np.array([(img.shape[0], img.shape[1]) for img in img_list],
             dtype = np.float32)
x = np.zeros((max_h, max_w, len(img_list), 1)).astype(np.float32)
for i, img in enumerate(img_list):
    x[:img.shape[0], :img.shape[1], i, 0] = (255.0 - img) / 255.0


model = Collapse(Model(1, [10, 10], collapse_type=['concat', 'sum']), collapse_type='sum')
loss  = Loss(model)
sgd   = SGD(loss, lr=0.0005)

# Round parameters for an easier visualization
for p in model.parameters:
    v = p.get_value()
    v = np.round(v, 1)
    p.set_value(v)

z = np.array([(max_h, max_w) for img in img_list],
             dtype = np.float32)

start = time.time()
for i in xrange(10000):
    output, sizes = model(x, z)
    l = sgd.train(x, z, y, ylen)
    if i % 200 == 0 or i == 9999:
        print 'ITER = %04d %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %8d' % (
            i, output.sum(), output.mean(), output.std(),
            output.min(), output.max(), l, time.time() - start)
        dec = decode(output, labels)
        print '{'
        print '  1: "%s"' % ''.join(dec[0])
        print '  2: "%s"' % ''.join(dec[1])
        print '}'
