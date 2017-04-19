import numpy as np
import theano
import theano.tensor as T
from scipy.ndimage import imread
from returnn import Model, Collapse, Loss, SGD

np.random.seed(1234)

H, W, N, K, D = 11, 7, 5, 3, 2
x = np.random.uniform(-1, 1, H * W * N * K).astype(np.float32).reshape((H, W, N, K))
x = np.round(x, 1)
z = np.array([
    [9,  5],
    [5,  7],
    [1,  7],
    [11, 1],
    [1,  1]
]).astype(np.float32)

model = Model(K, [D])
for p in model.parameters:
    v = p.get_value()
    v = np.round(v, 1)
    p.set_value(v)
    """
    print p.name
    if v.ndim == 1:
        print '  ',
        for i in xrange(v.shape[0]):
            print '% 3.1f,' % v[i],
        print ''
        print ''
    else:
        for i in xrange(v.shape[0]):
            print '  ',
            for j in xrange(v.shape[1]):
                print '% 3.1f,' % v[i][j],
            print ''
        print ''
    """

"""
for i in xrange(H):
    print '// y = %d' % (i)
    for j in xrange(W):
        for n in xrange(N):
            for k in xrange(K):
                print '% 3.1f,' % x[i,j,n,k],
        print ''
    print ''
"""

y = model(x, z)[0].reshape((H, W, N, D, 4))

for i in xrange(H):
    for j in xrange(W):
        print '// y = %d, x = %d' % (i, j)
        for n in xrange(N):
            for z in xrange(4):
                for d in xrange(D):
                    print '%-22s,' % float.hex(float(y[i,j,n,d,z])),
            print ''
        print ''
