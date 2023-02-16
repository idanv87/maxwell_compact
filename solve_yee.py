import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.linalg import polar

class Constants:
    DTYPE = tf.dtypes.float64

    PI = math.pi

    FILTER_BETA = tf.constant([[0., -1, 1, 0.], [0, 2, -2, 0], [0., -1, 1, 0.]],
                              shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_DELTA = tf.constant([[0., 0, 0, 0.], [-1, 3, -3, 1], [0., 0., 0., 0.]],
                               shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_GAMMA = tf.constant([[-1, 0, 0, 1], [0., 6, -6, 0.], [-1, 0., 0., 1]],
                               shape=[3, 4, 1, 1], dtype=DTYPE)
    FILTER_YEE = tf.constant([[0., 0, 0, 0.], [0, -1, 1, 0], [0., 0., 0., 0.]],
                             shape=[3, 4, 1, 1], dtype=DTYPE)

    # CENTRAL = tf.constant([-1, 1], shape=[1, 2, 1, 1], dtype=DTYPE)
    '''
    A is one sided derivative of order 4 kernel of 5 points used for the non-compact stencil
    we use
    '''
    A = np.array([-23 / 24, 7 / 8, 1 / 8, -1 / 24, 0.]).reshape(1, 5)

    def __init__(self, n, x, t, time_steps, k1_test, k2_test):
        self.XMAX = x
        self.N = n
        self.TIME_STEPS = time_steps
        self.T = t
        self.K1_TEST = k1_test
        self.K2_TEST = k2_test
        # self.TEST_NUM = len(self.K1_TEST) * len(self.K2_TEST)

        self.DT = self.T / (self.TIME_STEPS - 1)
        self.DX = self.XMAX / (self.N - 1)
        self.DY = self.XMAX / (self.N - 1)
        self.CFL = self.DT / self.DX
        self.X1 = np.linspace(0., self.XMAX, self.N)
        self.X2 = np.linspace(0., self.XMAX, self.N)
        self.X, self.Y = np.meshgrid(self.X1, self.X2, indexing='ij')
        '''
        4 tesnors below are padding kernels required to pad  rows/cols after covolution with 
        derivative
        '''
        self.PADX_FORWARD = tf.constant([[0, 0], [1, 1], [1, self.N - 2], [0, 0]], shape=[4, 2])
        self.PADX_BACWARD = tf.constant([[0, 0], [1, 1], [self.N - 2, 1], [0, 0]], shape=[4, 2])
        self.PADY_FORWARD = tf.constant([[0, 0], [1, self.N - 2], [1, 1], [0, 0]], shape=[4, 2])
        self.PADY_BACWARD = tf.constant([[0, 0], [self.N - 2, 1], [1, 1], [0, 0]], shape=[4, 2])

        '''
        4 tesnors below with the names "kernel_| are derivative kernels for the boundaries 
        which are added after the covolution is calculated in  the interior points
        (in tensor flow one cannot assign tensor to tensor and thats why we:
         1.calculate derivtive for inner point)
         2. pad with zeros 
         3. calculate the one sided derivative for 2 columns on each side (kernels forward and backwards) and pad
         4. calculate for two remaining rows with kernels fourth up and down (to avoid overlapping with step 3 we omit the 
         first two and last entries)
         two terms in each row so central derivative of order fourth is enough )
        '''
        self.B = np.zeros((1, self.N - 5 - 1))
        self.KERNEL_FORWARD = tf.cast(np.append(Constants.A, self.B).reshape(1, self.N - 1, 1, 1), Constants.DTYPE)
        self.KERNEL_BACKWARD = -tf.reverse(self.KERNEL_FORWARD, [1])
        self.PADEX_FORWARD = tf.constant([[0, 0], [0, 0], [0, self.N - 2], [0, 0]], shape=[4, 2])
        self.PADEX_BACKWARD = tf.constant([[0, 0], [0, 0], [self.N - 2, 0], [0, 0]], shape=[4, 2])
        self.PADEY_FORWARD = tf.constant([[0, 0], [0, self.N - 2], [0, 0], [0, 0]], shape=[4, 2])
        self.PADEY_BACKWARD = tf.constant([[0, 0], [self.N - 2, 0], [0, 0], [0, 0]], shape=[4, 2])
        self.D = np.zeros((1, self.N - 5))
        self.KERNEL_E_FORWARD = tf.cast(np.append(Constants.A, self.D).reshape(1, self.N, 1, 1), Constants.DTYPE)
        self.KERNEL_E_BACKWARD = -tf.reverse(self.KERNEL_E_FORWARD, [1])
        self.PADUP = tf.constant([[0, self.N - 3], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
        self.PADDOWN = tf.constant([[self.N - 3, 0], [0, 0], [0, 0], [0, 0]], shape=[4, 2])
        self.FOURTH_UP = tf.pad(
            tf.constant([1 / 24, -9 / 8, 9 / 8, -1 / 24], shape=[1, 4, 1, 1], dtype=Constants.DTYPE),
            self.PADUP)
        self.FOURTH_DOWN = tf.pad(
            tf.constant([1 / 24, -9 / 8, 9 / 8, -1 / 24], shape=[1, 4, 1, 1], dtype=Constants.DTYPE),
            self.PADDOWN)

        self.H = np.append(np.array([[35 / 16, -35 / 16, 21 / 16, -5 / 16]]), np.zeros((1, self.N - 5)))
        self.G = np.append(np.array([[4, -6, 4, -1]]), np.zeros((1, self.N - 6)))

        self.KLEFT = np.reshape(self.H, [1, self.H.shape[0], 1, 1])
        self.KRIGHT = tf.reverse(self.KLEFT, [1])

        self.KUP = np.reshape(self.G, [self.G.shape[0], 1, 1, 1])
        self.KDOWN = tf.reverse(self.KUP, [0])




def relative_norm(A, B, p=2):

    return tf.math.reduce_mean((abs(A-B)))

def loss_yee(beta, delta, gamma, test_data, C, norm='l2'):


    E = np.expand_dims(test_data['e'][0], axis=(0, -1))
    Hx = np.expand_dims(test_data['hx'][0], axis=(0, -1))
    Hy = np.expand_dims(test_data['hy'][0], axis=(0, -1))

    error = 0.
    for n in np.arange(0,C.TIME_STEPS-1,1):
    # for n in range(C.TIME_STEPS - 1):
        E = amper(E, Hx, Hy, beta, delta, gamma, C)
        Hx, Hy = faraday(E, Hx, Hy, beta, delta, gamma, C)
        #    plt.plot(E1[0,:,10,0],'-')
        #    plt.plot(test_data['e'][n + 1][:,10])
        #    plt.show()
        # print(q)

        if norm == 'l2':
            error += relative_norm(E[0, 1:-1, 1:-1, 0], test_data['e'][n + 1][1:-1,1:-1]) + \
                     relative_norm(Hx[0, 1:-1, :, 0], test_data['hx'][n + 1][1:-1,:]) + \
                     relative_norm(Hy[0, :, 1:-1, 0], test_data['hy'][n + 1][:,1:-1])
            error=error/(3 * (C.TIME_STEPS - 1))
        else:
            pass

    return error


def amper(E, Hx, Hy, beta, delta, gamma, C):
    cfl = C.CFL
    pad1 = pad_function([2, 2, 2, 2])
    pad5 = pad_function([C.N - 2, 1, 2, 2])
    pad6 = pad_function([2, 2, 1, C.N - 2])
    pad7 = pad_function([2, 2, C.N - 2, 1])
    pad4 = pad_function([1, C.N - 2, 2, 2])

    x1 = tf.math.multiply(beta, Dx(Hy, tf.transpose(C.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(delta, Dx(Hy, tf.transpose(C.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(Hy, tf.transpose(C.FILTER_YEE, perm=[1, 0, 2, 3]))
    x4 = tf.math.multiply(gamma, Dx(Hy, tf.transpose(C.FILTER_GAMMA, perm=[1, 0, 2, 3])))
    s1 = tf.pad(x1 + x2 + x3 + x4, pad1) + \
         tf.pad(Dx(Hy, tf.transpose(C.KERNEL_FORWARD, perm=[1, 0, 2, 3])), C.PADY_FORWARD) + \
         tf.pad(Dx(Hy, tf.transpose(C.KERNEL_BACKWARD, perm=[1, 0, 2, 3])), C.PADY_BACWARD) + \
         tf.pad(Dx(Hy, tf.transpose(C.FOURTH_UP, perm=[1, 0, 2, 3])), pad6) + \
         tf.pad(Dx(Hy, tf.transpose(C.FOURTH_DOWN, perm=[1, 0, 2, 3])), pad7)

    x1 = tf.math.multiply(beta, Dy(Hx, C.FILTER_BETA))
    x2 = tf.math.multiply(delta, Dy(Hx, C.FILTER_DELTA))
    x3 = Dy(Hx, C.FILTER_YEE)
    x4 = tf.math.multiply(gamma, Dy(Hx, C.FILTER_GAMMA))

    s2 = tf.pad(x1 + x2 + x3 + x4, pad1) + \
         tf.pad(Dy(Hx, C.KERNEL_FORWARD), C.PADX_FORWARD) + \
         tf.pad(Dy(Hx, C.KERNEL_BACKWARD), C.PADX_BACWARD) + \
         tf.pad(Dy(Hx, C.FOURTH_UP), pad4) + \
         tf.pad(Dy(Hx, C.FOURTH_DOWN), pad5)
    return E + (cfl) * (s1 - s2)


def faraday(E, Hx, Hy, beta, delta, gamma, C):
    cfl = C.CFL
    pad2 = pad_function([0, 0, 1, 1])
    pad3 = pad_function([1, 1, 0, 0])

    x1 = tf.math.multiply(beta, Dy(E, C.FILTER_BETA))
    x2 = tf.math.multiply(delta, Dy(E, C.FILTER_DELTA))
    x3 = Dy(E, C.FILTER_YEE)
    x4 = tf.math.multiply(gamma, Dy(E, C.FILTER_GAMMA))

    s3 = tf.pad(x1 + x2 + x3 + x4, pad2) + \
         tf.pad(Dy(E, C.KERNEL_E_FORWARD), C.PADEX_FORWARD)[:, 1:-1, :, :] + \
         tf.pad(Dy(E, C.KERNEL_E_BACKWARD), C.PADEX_BACKWARD)[:, 1:-1, :, :]

    x1 = tf.math.multiply(beta, Dx(E, tf.transpose(C.FILTER_BETA, perm=[1, 0, 2, 3])))
    x2 = tf.math.multiply(delta, Dx(E, tf.transpose(C.FILTER_DELTA, perm=[1, 0, 2, 3])))
    x3 = Dx(E, tf.transpose(C.FILTER_YEE, perm=[1, 0, 2, 3]))
    x4 = tf.math.multiply(gamma, Dx(E, tf.transpose(C.FILTER_GAMMA, perm=[1, 0, 2, 3])))

    s4 = tf.pad(x1 + x2 + x3 + x4, pad3) + \
         tf.pad(Dx(E, tf.transpose(C.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])),
                C.PADEY_FORWARD)[:, :, 1:-1, :] + \
         tf.pad(Dx(E, tf.transpose(C.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])),
                C.PADEY_BACKWARD)[:, :, 1:-1, :]

    return Hx - (cfl) * s3, Hy + (cfl) * s4


def Dy(B, kernel):
    return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


def Dx(B, kernel):
    return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


def pad_function(input):
    return tf.constant([[0, 0], [input[0], input[1]], [input[2], input[3]], [0, 0]], shape=[4, 2])

