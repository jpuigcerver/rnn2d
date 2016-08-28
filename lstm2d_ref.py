import numpy as np
#import theano as T
import struct

DTYPE = np.float32
#T.config.floatX = 'float32'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x):
    return x


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

def printQ(k, y, x, a, g_i, g_o, g_fy, g_fx, c):
    N = a.shape[0]
    for n in xrange(N):
        print 'Q(k=%d,y=%d,x=%d,n=%d,g=0) = %s' % (k, y, x, n, a[n,:])
        print 'Q(k=%d,y=%d,x=%d,n=%d,g=1) = %s' % (k, y, x, n, g_i[n,:])
        print 'Q(k=%d,y=%d,x=%d,n=%d,g=2) = %s' % (k, y, x, n, g_o[n,:])
        print 'Q(k=%d,y=%d,x=%d,n=%d,g=3) = %s' % (k, y, x, n, g_fy[n,:])
        print 'Q(k=%d,y=%d,x=%d,n=%d,g=4) = %s' % (k, y, x, n, g_fx[n,:])
        print 'Q(k=%d,y=%d,x=%d,n=%d,g=5) = %s' % (k, y, x, n, c[n,:])
        print ''

def lstm_2d_theano(X, dO, w, ry, rx, b,
                   f_g = lambda x: x, f_i = lambda x: x, f_o = lambda x: x):
    H, W, N, K = X.shape
    D = b.shape[-1]
    assert(b.shape  == (4, 5, D))
    assert(w.shape  == (4, K, 5, D))
    assert(ry.shape == (4, D, 5, D))
    assert(rx.shape == (4, D, 5, D))
    X  = T.shared(X, name='X')
    b  = T.shared(b, name='b')
    w  = T.shared(w, name='w')
    ry = T.shared(ry, name='ry')
    rx = T.shared(rx, name='rx')
    Ot  = [[[None for x in xrange(W)] for y in xrange(H)] for z in xrange(4)]
    Qt  = [[[[None for g in xrange(6)]for x in xrange(W)] for y in xrange(H)] \
           for z in xrange(4)]
    #dQ  = [[[[None for g in xrange(6)]for x in xrange(W)] for y in xrange(H)] \
    #       for z in xrange(4)]

    #O  = T.tensor.zeros((H, W, N, 4, D))
    #Q  = #T.tensor.zeros((4, H, W, N, 5, D))
    #O = T.shared(np.zeros((H, W, N, 4, D)), 'O')
    #Q = T.shared(np.zeros((4, H, W, N, 6, D)), 'Q')
    for z in xrange(4):
        for i in xrange(H):
            for j in xrange(W):
                y = i if z == 0 or z == 1 else H - i - 1
                x = j if z == 0 or z == 2 else W - j - 1
                Qt[z][y][x][0] = \
                    b[z,0,:] + T.tensor.dot(X[y,x,:,:], w[z,:,0,:])
                Qt[z][y][x][1] = \
                    b[z,1,:] + T.tensor.dot(X[y,x,:,:], w[z,:,1,:])
                Qt[z][y][x][2] = \
                    b[z,2,:] + T.tensor.dot(X[y,x,:,:], w[z,:,2,:])
                Qt[z][y][x][3] = \
                    b[z,3,:] + T.tensor.dot(X[y,x,:,:], w[z,:,3,:])
                Qt[z][y][x][4] = \
                    b[z,4,:] + T.tensor.dot(X[y,x,:,:], w[z,:,4,:])

                yp = y - 1 if z == 0 or z == 1 else y + 1
                if yp >= 0 and yp < H:
                    Qt[z][y][x][0] += T.tensor.dot(Ot[z][yp][x], ry[z,:,0,:])
                    Qt[z][y][x][1] += T.tensor.dot(Ot[z][yp][x], ry[z,:,1,:])
                    Qt[z][y][x][2] += T.tensor.dot(Ot[z][yp][x], ry[z,:,2,:])
                    Qt[z][y][x][3] += T.tensor.dot(Ot[z][yp][x], ry[z,:,3,:])
                    Qt[z][y][x][4] += T.tensor.dot(Ot[z][yp][x], ry[z,:,4,:])
                xp = x - 1 if z == 0 or z == 2 else x + 1
                if xp >= 0 and xp < W:
                    Qt[z][y][x][0] += T.tensor.dot(Ot[z][y][xp], rx[z,:,0,:])
                    Qt[z][y][x][1] += T.tensor.dot(Ot[z][y][xp], rx[z,:,1,:])
                    Qt[z][y][x][2] += T.tensor.dot(Ot[z][y][xp], rx[z,:,2,:])
                    Qt[z][y][x][3] += T.tensor.dot(Ot[z][y][xp], rx[z,:,3,:])
                    Qt[z][y][x][4] += T.tensor.dot(Ot[z][y][xp], rx[z,:,4,:])

                Qt[z][y][x][5] = f_i(Qt[z][y][x][0]) * f_g(Qt[z][y][x][1])
                if yp >= 0 and yp < H:
                    Qt[z][y][x][5] += f_g(Qt[z][y][x][3]) * Qt[z][yp][x][5]
                if xp >= 0 and xp < W:
                    Qt[z][y][x][5] += f_g(Qt[z][y][x][4]) * Qt[z][y][xp][5]
                Ot[z][y][x] = f_o(Qt[z][y][x][5]) * f_g(Qt[z][y][x][2])

    J = None
    for z in xrange(4):
        for y in xrange(H):
            for x in xrange(W):
                t = (Ot[z][y][x] * dO[y,x,:,z,:]).sum()
                J = J + t if J else t

    O = np.zeros((H, W, N, 4, D), dtype=DTYPE)
    for z in xrange(4):
        for y in xrange(H):
            for x in xrange(W):
                O[y,x,:,z,:] = Ot[z][y][x].eval()

    return O, J.eval(), T.grad(J, X).eval(), T.grad(J, b).eval(), \
        T.grad(J, w).eval(), T.grad(J, ry).eval(), T.grad(J, rx).eval()

def lstm_2d(X, b, w, ry, rx, f_g = linear, f_i = linear, f_o = linear):
    H, W, N, K = X.shape
    D = b.shape[-1]
    assert(b.shape  == (4, 5, D))
    assert(w.shape  == (4, K, 5, D))
    assert(ry.shape == (4, D, 5, D))
    assert(rx.shape == (4, D, 5, D))
    O = np.zeros((H, W, N, 4, D), dtype = DTYPE)
    Q = np.zeros((4, H, W, N, 6, D), dtype = DTYPE)
    for z in xrange(4):
        for i in xrange(H):
            for j in xrange(W):
                y = i if z == 0 or z == 1 else H - i - 1
                x = j if z == 0 or z == 2 else W - j - 1
                Q[z,y,x,:,0,:] = \
                    np.copy(b[z,0,:]) + np.dot(X[y,x,:,:], w[z,:,0,:])
                Q[z,y,x,:,1,:] = \
                    np.copy(b[z,1,:]) + np.dot(X[y,x,:,:], w[z,:,1,:])
                Q[z,y,x,:,2,:] = \
                    np.copy(b[z,2,:]) + np.dot(X[y,x,:,:], w[z,:,2,:])
                Q[z,y,x,:,3,:] = \
                    np.copy(b[z,3,:]) + np.dot(X[y,x,:,:], w[z,:,3,:])
                Q[z,y,x,:,4,:] = \
                    np.copy(b[z,4,:]) + np.dot(X[y,x,:,:], w[z,:,4,:])

                yp = y - 1 if z == 0 or z == 1 else y + 1
                if yp >= 0 and yp < H:
                    Q[z,y,x,:,0,:] += np.dot(O[yp,x,:,z,:], ry[z,:,0,:])
                    Q[z,y,x,:,1,:] += np.dot(O[yp,x,:,z,:], ry[z,:,1,:])
                    Q[z,y,x,:,2,:] += np.dot(O[yp,x,:,z,:], ry[z,:,2,:])
                    Q[z,y,x,:,3,:] += np.dot(O[yp,x,:,z,:], ry[z,:,3,:])
                    Q[z,y,x,:,4,:] += np.dot(O[yp,x,:,z,:], ry[z,:,4,:])
                xp = x - 1 if z == 0 or z == 2 else x + 1
                if xp >= 0 and xp < W:
                    Q[z,y,x,:,0,:] += np.dot(O[y,xp,:,z,:], rx[z,:,0,:])
                    Q[z,y,x,:,1,:] += np.dot(O[y,xp,:,z,:], rx[z,:,1,:])
                    Q[z,y,x,:,2,:] += np.dot(O[y,xp,:,z,:], rx[z,:,2,:])
                    Q[z,y,x,:,3,:] += np.dot(O[y,xp,:,z,:], rx[z,:,3,:])
                    Q[z,y,x,:,4,:] += np.dot(O[y,xp,:,z,:], rx[z,:,4,:])

                Q[z,y,x,:,5,:] = f_i(Q[z,y,x,:,0,:]) * f_g(Q[z,y,x,:,1,:])
                if yp >= 0 and yp < H:
                    Q[z,y,x,:,5,:] += f_g(Q[z,y,x,:,3,:]) * Q[z,yp,x,:,5,:]
                if xp >= 0 and xp < W:
                    Q[z,y,x,:,5,:] += f_g(Q[z,y,x,:,4,:]) * Q[z,y,xp,:,5,:]
                O[y,x,:,z,:] = f_o(Q[z,y,x,:,5,:]) * f_g(Q[z,y,x,:,2,:])
    return O, Q

def lstm_2d_bw(X, Q, dO, b, w, ry, rx,
               f_g = linear, f_i = linear, f_o = linear,
               fp_g = linear, fp_i = linear, fp_o = linear):
    H, W, N, K = X.shape
    D = b.shape[-1]
    assert(b.shape  == (4, 5, D))
    assert(w.shape  == (4, K, 5, D))
    assert(ry.shape == (4, D, 5, D))
    assert(rx.shape == (4, D, 5, D))
    dX = np.zeros((H, W, N, K), dtype = DTYPE)
    dQ = np.zeros((4, H, W, N, 6, D), dtype = DTYPE)
    for z in xrange(4):
        for i in xrange(H):
            for j in xrange(W):
                y = H - i - 1 if z == 0 or z == 1 else i
                x = W - j - 1 if z == 0 or z == 2 else j
                yp = y - 1 if z == 0 or z == 1 else y + 1
                xp = x - 1 if z == 0 or z == 2 else x + 1
                yn = y + 1 if z == 0 or z == 1 else y - 1
                xn = x + 1 if z == 0 or z == 2 else x - 1
                ## dJ/dC(y,x)
                dQ[z,y,x,:,5,:] = \
                    dO[y,x,:,z,:] * f_g(Q[z,y,x,:,2,:]) * fp_o(Q[z,y,x,:,5,:])
                if yn >= 0 and yn < H:
                    dQ[z,y,x,:,5,:] += dQ[z,yn,x,:,5,:] * f_g(Q[z,yn,x,:,3,:])
                if xn >= 0 and xn < W:
                    dQ[z,y,x,:,5,:] += dQ[z,y,xn,:,5,:] * f_g(Q[z,y,xn,:,4,:])
                ## dJ/dgfx(y,x)
                dQ[z,y,x,:,4,:] = \
                    dQ[z,y,x,:,5,:] * Q[z,y,xp,:,5,:] * fp_g(Q[z,y,x,:,4,:]) \
                    if xp >= 0 and xp < W else 0
                ## dJ/dgfy(y,x)
                dQ[z,y,x,:,3,:] = \
                    dQ[z,y,x,:,5,:] * Q[z,yp,x,:,5,:] * fp_g(Q[z,y,x,:,3,:]) \
                    if yp >= 0 and yp < H else 0
                ## dJ/dgo(y,x)
                dQ[z,y,x,:,2,:] = \
                    dO[y,x,:,z,:]   * f_o(Q[z,y,x,:,5,:]) * fp_g(Q[z,y,x,:,2,:])
                ## dJ/dgi(y,x)
                dQ[z,y,x,:,1,:] = \
                    dQ[z,y,x,:,5,:] * f_i(Q[z,y,x,:,0,:]) * fp_g(Q[z,y,x,:,1,:])
                ## dJ/dX(y,x)
                dX[y,x,:,:] += np.dot(dQ[z,y,x,:,0,:], w[z,:,0,:].transpose())
                dX[y,x,:,:] += np.dot(dQ[z,y,x,:,1,:], w[z,:,1,:].transpose())
                dX[y,x,:,:] += np.dot(dQ[z,y,x,:,2,:], w[z,:,2,:].transpose())
                dX[y,x,:,:] += np.dot(dQ[z,y,x,:,3,:], w[z,:,3,:].transpose())
                dX[y,x,:,:] += np.dot(dQ[z,y,x,:,4,:], w[z,:,4,:].transpose())
    return dX

if __name__ == '__main__':
    H, W, N, K, D = 2, 3, 2, 3, 2
    #X = np.random.randint(0, 100, size=(H, W, N, K)) / 100.0
    #b = np.random.randint(-100, 100, size=(4, 5, D)) / 100.0
    #w = np.random.randint(-100, 100, size=(4, K, 5, D)) / 100.0
    #ry = np.random.randint(-100, 100, size=(4, D, 5, D)) / 100.0
    #rx = np.random.randint(-100, 100, size=(4, D, 5, D)) / 100.0
    ## Input initialization (shape: (H, W, N, K))
    X = np.asarray([
        [
            [[0.30, 0.68, 0.29],
             [0.10, 0.70, 0.88]],
            [[0.13, 0.18, 0.35],
             [0.86, 0.66, 0.75]],
            [[0.53, 0.40, 0.48],
             [0.20, 0.58, 0.66]]
        ],
        [
            [[0.30, 0.99, 0.64],
             [0.46, 0.44, 0.65]],
            [[0.82, 0.59, 0.47],
             [0.18, 0.53, 0.13]],
            [[0.68, 0.79, 0.80],
             [0.32, 0.09, 0.40]]
        ]
    ], dtype = DTYPE)
    ## Bias initialization. Note: all directions use the same bias
    b = np.zeros((4, 5, D), dtype = DTYPE)
    b[0,:,:] = np.asarray([
        [-0.66, -0.56],
        [-0.19,  0.94],
        [-0.22,  0.12],
        [-0.99, -0.08],
        [-0.79, -0.69]
    ])
    b[3,:,:] = b[2,:,:] = b[1,:,:] = b[0,:,:]
    ## Input weights initialization. Note: all directions use the same weights
    w = np.zeros((4, K, 5, D), dtype = DTYPE)
    w[0,:,:,:] = np.asarray([
        [[-0.69,  0.16],
         [ 0.64,  0.11],
         [ 0.72, -0.48],
         [ 0.67,  0.72],
         [ 0.61, -0.34]],

        [[ 0.72, -0.39],
         [-0.31, -0.76],
         [ 0.31, -0.88],
         [ 0.24, -0.50],
         [-0.65, -0.21]],

        [[ 0.61,  0.95],
         [-0.46,  0.79],
         [ 0.98, -0.89],
         [ 0.88,  0.62],
         [-0.36,  0.07]]
    ])
    w[3,:,:,:] = w[2,:,:,:] = w[1,:,:,:] = w[0,:,:,:]
    ry = np.zeros((4, D, 5, D), dtype = DTYPE)
    ry[0,:,:,:] = np.asarray([
        [[ 0.16,  0.10],
         [ 0.01,  0.91],
         [-0.05,  0.38],
         [ 0.38, -0.62],
         [ 0.99, -0.03]],

        [[ 0.60,  0.30],
         [-0.47, -0.03],
         [ 0.12, -0.77],
         [ 0.94,  0.77],
         [-0.79,  0.76]]
    ])
    ry[3,:,:,:] = ry[2,:,:,:] = ry[1,:,:,:] = ry[0,:,:,:]
    rx = np.zeros((4, D, 5, D), dtype = DTYPE)
    rx[0,:,:,:] = np.asarray([
        [[-0.30, -0.80],
         [ 0.93,  0.90],
         [ 0.95, -0.50],
         [ 0.65,  0.23],
         [-0.90,  0.36]],

        [[-0.42,  0.39],
         [ 0.54, -0.20],
         [ 0.14, -0.16],
         [ 0.57,  0.51],
         [-0.30,  0.88]]
    ])
    rx[3,:,:,:] = rx[2,:,:,:] = rx[1,:,:,:] = rx[0,:,:,:]
    #dO = np.random.choice([-1, 1, 1, 1], size=(H, W, N, 4, D)) * np.random.randint(0,100,size=(H, W, N, 4, D)) / 100.0
    dO = np.asarray([
        [[[[ 0.51,  0.  ],
           [ 0.21,  0.47],
           [-0.06,  0.26],
           [ 0.5,  -0.71]],
          [[ 0.53,  0.65],
           [ 0.52,  0.25],
           [-0.39, -0.13],
           [ 0.05,  0.07]]],
         [[[ 0.44,  0.66],
           [ 0.3,   0.98],
           [ 0.2,   0.76],
           [-0.93,  0.42]],
          [[ 0.17,  0.71],
           [ 0.16, -0.48],
           [ 0.39,  0.92],
           [ 0.04,  0.81]]],
         [[[ 0.07,  0.98],
           [-0.17,  0.79],
           [ 0.57,  0.39],
           [ 0.94,  0.4 ]],
          [[ 0.81,  0.4 ],
           [ 0.81,  0.34],
           [ 0.74,  0.49],
           [ 0.68,  0.  ]]]],

        [[[[ 0.29,  0.29],
           [ 0.5,   0.52],
           [-0.15, -0.63],
           [-0.87,  0.43]],
          [[ 0.39,  0.59],
           [-0.68,  0.92],
           [ 0.43, -0.16],
           [-0.27,  0.19]]],
         [[[-0.84,  0.13],
           [ 0.33,  0.89],
           [-0.47,  0.72],
           [-0.47,  0.27]],
          [[ 0.85, -0.23],
           [ 0.15, -0.61],
           [ 0.69,  0.76],
           [ 0.47,  0.56]]],
         [[[ 0.13,  0.61],
           [ 0.71,  0.11],
           [-0.44,  0.11],
           [ 0.47,  0.04]],
          [[ 0.,    0.78],
           [ 0.8,   0.24],
           [ 0.4,   0.49],
           [-0.93,  0.09]]]]
    ])

    O, Q = lstm_2d(X, b, w, ry, rx)
    dX = lstm_2d_bw(X, Q, dO, b, w, ry, rx)

    '''
    O, J, dX, db, dw, dry, drx = lstm_2d_theano(X, dO, w, ry, rx, b)

    print 'J =', J
    print 'O ='
    for y in xrange(H):
        for x in xrange(W):
            for n in xrange(N):
                for z in xrange(4):
                    for d in xrange(D):
                        print ('%s,' % float_to_hex(O[y, x, n, z, d])),
                print ''
    '''
    print 'dJ/dX ='
    for y in xrange(H):
        for x in xrange(W):
            for n in xrange(N):
                for k in xrange(K):
                    print ('%-.4f,' % dX[y, x, n, k]),
                print ''
    '''
    print 'dJ/db ='
    for z in xrange(4):
        for g in xrange(5):
            for d in xrange(D):
                print ('%-.4f,' % db[z, g, d]),
            print ''
    print 'dJ/dw ='
    for z in xrange(4):
        for d1 in xrange(K):
            for g in xrange(5):
                for d2 in xrange(D):
                    print ('%-.4f,' % dw[z, d1, g, d2]),
                print ''
    print 'dJ/dry ='
    for z in xrange(4):
        for d1 in xrange(D):
            for g in xrange(5):
                for d2 in xrange(D):
                    print ('%-.4f,' % dry[z, d1, g, d2]),
                print ''
    print 'dJ/drx ='
    for z in xrange(4):
        for d1 in xrange(D):
            for g in xrange(5):
                for d2 in xrange(D):
                    print ('%-.4f,' % drx[z, d1, g, d2]),
                print ''
    '''
