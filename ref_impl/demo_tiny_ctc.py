import numpy as np
import theano
import theano.tensor as T
from scipy.ndimage import imread
from returnn import Model, Collapse, Loss, SGD

np.random.seed(1234)

img_list = [ imread('../assets/a.png').astype(np.float32),
             imread('../assets/ab.png').astype(np.float32) ]
y    = np.array([1, 1, 2], dtype=np.int32)
ylen = np.array([1, 2], dtype=np.int32)

max_h, max_w, max_c = 0, 0, 0
for img in img_list:
    max_h = max(max_h, img.shape[0])
    max_w = max(max_w, img.shape[1])
    max_c = max(max_c, img.shape[2])

z = np.array([(img.shape[0], img.shape[1]) for img in img_list],
             dtype = np.float32)
x = np.zeros((max_h, max_w, len(img_list), max_c)).astype(np.float32)
for i, img in enumerate(img_list):
    x[:img.shape[0], :img.shape[1], i, :img.shape[2]] = (255.0 - img) / 255.0


model = Collapse(Model(max_c, [5], collapse_type='sum'), collapse_type='sum')
loss  = Loss(model)
sgd   = SGD(loss)

# Round parameters for an easier visualization
for p in model.parameters:
    v = p.get_value()
    v = np.round(v, 1)
    p.set_value(v)

z = np.array([(max_h, max_w) for img in img_list],
             dtype = np.float32)

for i in xrange(20):
    output, sizes = model(x, z)
    l = sgd.train(x, z, y, ylen)
    print 'ITER = %02d %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f' % (
        i, output.sum(), output.mean(), output.std(),
        output.min(), output.max(), l)
