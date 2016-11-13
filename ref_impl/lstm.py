import numpy as np
import theano as T
import struct

DTYPE = np.float64
T.config.floatX = 'float64'
hex_dump = lambda x: double_to_hex(x)

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

def lstm_2d_theano(X, S, dO, w, ry, rx, b,
                   f_g = lambda x: 1 / (1 + T.tensor.exp(-x)),
                   f_i = lambda x: T.tensor.tanh(x),
                   f_o = lambda x: T.tensor.tanh(x)):
    H, W, N, K = X.shape
    D = b.shape[-1]
    assert(b.shape  == (4, 5, D))
    assert(w.shape  == (4, K, 5, D))
    assert(ry.shape == (4, D, 5, D))
    assert(rx.shape == (4, D, 5, D))
    X  = X.astype(DTYPE)
    dO = dO.astype(DTYPE)
    w  = w.astype(DTYPE)
    ry = ry.astype(DTYPE)
    rx = rx.astype(DTYPE)
    b  = b.astype(DTYPE)
    X  = T.shared(X, name='X')
    b  = T.shared(b, name='b')
    w  = T.shared(w, name='w')
    ry = T.shared(ry, name='ry')
    rx = T.shared(rx, name='rx')
    Ot = [[[[None for n in xrange(N)]
            for x in xrange(W)] for y in xrange(H)] for z in xrange(4)]
    Qt = [[[[[None for g in xrange(6)] for n in xrange(N)] \
            for x in xrange(W)] for y in xrange(H)] for z in xrange(4)]
    for z in xrange(4):
        for i in xrange(H):
            for j in xrange(W):
                y = i if z == 0 or z == 1 else H - i - 1
                x = j if z == 0 or z == 2 else W - j - 1
                for n in xrange(N):
                    if y < S[n][0] and x < S[n][1]:
                        Qt[z][y][x][n][0] = \
                            b[z,0,:] + T.tensor.dot(X[y,x,n,:], w[z,:,0,:])
                        Qt[z][y][x][n][1] = \
                            b[z,1,:] + T.tensor.dot(X[y,x,n,:], w[z,:,1,:])
                        Qt[z][y][x][n][2] = \
                            b[z,2,:] + T.tensor.dot(X[y,x,n,:], w[z,:,2,:])
                        Qt[z][y][x][n][3] = \
                            b[z,3,:] + T.tensor.dot(X[y,x,n,:], w[z,:,3,:])
                        Qt[z][y][x][n][4] = \
                            b[z,4,:] + T.tensor.dot(X[y,x,n,:], w[z,:,4,:])

                        yp = y - 1 if z == 0 or z == 1 else y + 1
                        if yp >= 0 and yp < S[n][0]:
                            Qt[z][y][x][n][0] += \
                                T.tensor.dot(Ot[z][yp][x][n], ry[z,:,0,:])
                            Qt[z][y][x][n][1] += \
                                T.tensor.dot(Ot[z][yp][x][n], ry[z,:,1,:])
                            Qt[z][y][x][n][2] += \
                                T.tensor.dot(Ot[z][yp][x][n], ry[z,:,2,:])
                            Qt[z][y][x][n][3] += \
                                T.tensor.dot(Ot[z][yp][x][n], ry[z,:,3,:])
                            Qt[z][y][x][n][4] += \
                                T.tensor.dot(Ot[z][yp][x][n], ry[z,:,4,:])
                        xp = x - 1 if z == 0 or z == 2 else x + 1
                        if xp >= 0 and xp < S[n][1]:
                            Qt[z][y][x][n][0] += \
                                T.tensor.dot(Ot[z][y][xp][n], rx[z,:,0,:])
                            Qt[z][y][x][n][1] += \
                                T.tensor.dot(Ot[z][y][xp][n], rx[z,:,1,:])
                            Qt[z][y][x][n][2] += \
                                T.tensor.dot(Ot[z][y][xp][n], rx[z,:,2,:])
                            Qt[z][y][x][n][3] += \
                                T.tensor.dot(Ot[z][y][xp][n], rx[z,:,3,:])
                            Qt[z][y][x][n][4] += \
                                T.tensor.dot(Ot[z][y][xp][n], rx[z,:,4,:])

                        Qt[z][y][x][n][5] = \
                            f_i(Qt[z][y][x][n][0]) * f_g(Qt[z][y][x][n][1])
                        if yp >= 0 and yp < S[n][0]:
                            Qt[z][y][x][n][5] += \
                                f_g(Qt[z][y][x][n][3]) * Qt[z][yp][x][n][5]
                        if xp >= 0 and xp < S[n][1]:
                            Qt[z][y][x][n][5] += \
                                f_g(Qt[z][y][x][n][4]) * Qt[z][y][xp][n][5]

                        Ot[z][y][x][n] = \
                            f_o(Qt[z][y][x][n][5]) * f_g(Qt[z][y][x][n][2])
                    else:
                        Qt[z][y][x][n][0] = T.tensor.zeros((1, D))
                        Qt[z][y][x][n][1] = T.tensor.zeros((1, D))
                        Qt[z][y][x][n][2] = T.tensor.zeros((1, D))
                        Qt[z][y][x][n][3] = T.tensor.zeros((1, D))
                        Qt[z][y][x][n][4] = T.tensor.zeros((1, D))
                        Qt[z][y][x][n][5] = T.tensor.zeros((1, D))
                        Ot[z][y][x][n] = T.tensor.zeros((1, D))

    J = None
    for z in xrange(4):
        for y in xrange(H):
            for x in xrange(W):
                for n in xrange(N):
                    if y < S[n][0] and x < S[n][1]:
                        t = (Ot[z][y][x][n] * dO[y,x,n,z,:]).sum()
                        J = J + t if J else t

    O = np.zeros((H, W, N, 4, D), dtype=DTYPE)
    for z in xrange(4):
        for y in xrange(H):
            for x in xrange(W):
                for n in xrange(N):
                    if y < S[n][0] and x < S[n][1]:
                        O[y,x,n,z,:] = Ot[z][y][x][n].eval()
                    else:
                        O[y,x,n,z,:] = 0

    return J.eval(), O, T.grad(J, X).eval(), T.grad(J, b).eval(), \
        T.grad(J, w).eval(), T.grad(J, ry).eval(), T.grad(J, rx).eval()

def print_X(X, H, W, N, K, name='X'):
    print '%s = {' % name
    for y in xrange(H):
        for x in xrange(W):
            print '// %s(y = %d, x = %d)' % (name, y, x)
            for n in xrange(N):
                for k in xrange(K):
                    print '%5.2f,' % X[y,x,n,k],
                print ''
    print '}'

def print_O(O, H, W, N, D, name='O'):
    print '%s = {' % name
    for y in xrange(H):
        for x in xrange(W):
            print '// %s(y = %d, x = %d)' % (name, y, x)
            for n in xrange(N):
                for z in xrange(4):
                    for d in xrange(D):
                        print '%5.2f,' % O[y,x,n,z,d],
                print ''
    print '}'

def print_P(b, w, ry, rx, K, D, name='P'):
    print '%s = {' % name
    for z in xrange(4):
        print '// bias, z = %d' % z
        for g in xrange(5):
            for d in xrange(D):
                print '%5.2f,' % b[z,g,d],
        print ''
        print '// input weight, z = %d' % z
        for k in xrange(K):
            for g in xrange(5):
                for d in xrange(D):
                    print '%5.2f,' % w[z,k,g,d],
            print ''
        print '// recurrent-y weight, z = %d' % z
        for d1 in xrange(D):
            for g in xrange(5):
                for d2 in xrange(D):
                    print '%5.2f,' % ry[z,d1,g,d2],
            print ''
        print '// recurrent-x weight, z = %d' % z
        for d1 in xrange(D):
            for g in xrange(5):
                for d2 in xrange(D):
                    print '%5.2f,' % rx[z,d1,g,d2],
            print ''
    print '}'

if __name__ == '__main__':
    np.random.seed(0)

    H, W, N, K, D = 3, 4, 2, 3, 2
    S = [(2, 4), (3, 3)]
    X = np.random.randint(0, 100, size=(H, W, N, K)) / 100.0
    b = np.random.randint(-100, 100, size=(4, 5, D)) / 100.0
    w = np.random.randint(-100, 100, size=(4, K, 5, D)) / 100.0
    ry = np.random.randint(-100, 100, size=(4, D, 5, D)) / 100.0
    rx = np.random.randint(-100, 100, size=(4, D, 5, D)) / 100.0
    dO = np.random.choice([-1, 1, 1, 1], size=(H, W, N, 4, D)) * \
         np.random.randint(0,100,size=(H, W, N, 4, D)) / 100.0

    # Put zeros to padded zones
    X[2:,:,0,:] = 0
    X[:,4:,0,:] = 0
    X[3:,:,1,:] = 0
    X[:,3:,1,:] = 0

    # Put zeros to padded zones
    dO[2:,:,0,:,:] = 0
    dO[:,4:,0,:,:] = 0
    dO[3:,:,1,:,:] = 0
    dO[:,3:,1,:,:] = 0

    print_X(X, H, W, N, K, 'I')
    print_O(dO, H, W, N, D, 'dO')
    print_P(b, w, ry, rx, K, D, 'P')

    J, O, dX, db, dw, dry, drx = lstm_2d_theano(X, S, dO, w, ry, rx, b)

    sum_O  = np.sum(O)
    sum_dX = np.sum(dX)
    sum_dP = np.sum(db) + np.sum(dw) + np.sum(dry) + np.sum(drx)

    print 'J =', J
    print 'sum O = %.32f | %s' % (sum_O, hex_dump(sum_O))
    print 'sum dX = %.32f | %s' % (sum_dX, hex_dump(sum_dX))
    print 'sum dP = %.32f | %s' % (sum_dP, hex_dump(sum_dP))

    exit(0)
