#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import numpy as np
import sys
import os
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from extended_kalman_filter import *
from unscented_kalman_filter import *
from unscented_kalman_filter_rungekutta import *


print('非線形カルマンフィルター実行時間比較')
np.random.seed(1024)

def set_kalman_filter_functions(dt):
    '''
     周回運動モデル
    '''

    def f(x, u=None):
        return np.mat([\
            [x[1,0]], \
            [-x[0,0]], \
            [x[3,0]], \
            [-x[2,0]], \
            [0.0], \
            [0.0]\
        ])

    def fd(x, u=None):
        return np.mat([\
            [x[0,0]+dt*x[1,0]], \
            [x[1,0]-dt*x[0,0]], \
            [x[2,0]+dt*x[3,0]], \
            [x[3,0]-dt*x[2,0]], \
            [x[4,0]], \
            [x[5,0]]\
        ])

    def h(x, u=None):
        return np.mat([\
            [math.sqrt((x[0,0]-x[4,0])**2+(x[2,0]-x[5,0])**2)], \
            [math.atan((x[2,0]-x[5,0])/(x[0,0]-x[4,0]))]\
        ])

    def F(x, u=None):
        return np.mat([\
            [1.0 , dt, 0.0, 0.0, 0.0, 0.0], \
            [-dt, 1.0, 0.0, 0.0, 0.0, .0], \
            [0.0, 0.0, 1.0 , dt, 0.0, 0.0], \
            [0.0, 0.0, -dt, 1.0, 0.0, 0.0], \
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], \
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] \
        ])

    def H(x=None, u=None):
        return np.mat([\
            [\
                (x[0,0]-x[4,0])/math.sqrt((x[0,0]-x[4,0])**2+(x[2,0]-x[5,0])**2),\
                0.0,\
                (x[2,0]-x[5,0])/math.sqrt((x[0,0]-x[4,0])**2+(x[2,0]-x[5,0])**2),\
                0.0,\
                -(x[0,0]-x[4,0])/math.sqrt((x[0,0]-x[4,0])**2+(x[2,0]-x[5,0])**2),\
                -(x[2,0]-x[5,0])/math.sqrt((x[0,0]-x[4,0])**2+(x[2,0]-x[5,0])**2)
            ], \
            [\
                (-(x[2,0]-x[5,0])/((x[0,0]-x[4,0])**2)/(1+((x[2,0]-x[5,0]))/(x[0,0]-x[4,0]))**2),\
                0.0,\
                (1.0/(x[0,0]-x[4,0]))/(1.0+((x[2,0]-x[5,0])/(x[0,0]-x[4,0]))**2),\
                0.0,\
                ((x[2,0]-x[5,0])/((x[0,0]-x[4,0])**2))/(1+((x[2,0]-x[5,0])/(x[0,0]-x[4,0]))**2),\
                (-1.0/(x[0,0]-x[4,0]))/(1.0+((x[2,0]-x[5,0])/(x[0,0]-x[4,0]))**2),\
            ]\
        ])

    return f, fd, h, F, H

calculate_sec = 0.1
dt = 0.01
step = int(calculate_sec / dt)
T = np.arange(step+1)

x_dim = 6
y_dim = 2
u_dim = 1

x0 = np.mat([[1.5],[0.0],[0.0],[1.5],[5.0],[3.0]])
U = np.mat([range(step+1)])

Q = np.mat([[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0]])
R = np.mat([[0.05,0],[0,0.05]])

(f, fd, h, F, H) = set_kalman_filter_functions(dt)

x0_estimate = np.mat([[0.01],[0.01],[0.01],[0.01],[0.1],[0.1]])
P0 = np.mat([[1.0,0,0,0,0,0],[0,1.0,0,0,0,0],[0,0,1.0,0,0,0],[0,0,0,1.0,0,0],[0,0,0,0,0.1,0],[0,0,0,0,0,0.1]])

X_noise = np.mat(np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, step+1)).T
Y_noise = np.mat(np.random.multivariate_normal(np.zeros(R.shape[0]), R, step+1)).T
X_state = np.mat(np.zeros((x_dim,step+1)))
X_state[:,0] = x0 + X_noise[:,0]
X_observed = np.mat(np.zeros((y_dim,step+1)))
X_observed[:,0] = h(X_state[:,0]) + Y_noise[:,0]

ekf = ExtendedKalmanFilter(fd, h, F, H, Q, R, x0_estimate, P0, u_dim, step)
ukf = UnscentedKalmanFilter(fd, h, Q, R, x0_estimate, P0, u_dim, step, 0)
ukfr = UnscentedKalmanFilterRungeKutta(f, h, Q, R, x0_estimate, P0, u_dim, step, dt, 0)


for i in range(0,step):
    uk = U[:,i]
    X_state[:,i+1] = UnscentedKalmanFilterRungeKutta.runge_kutta(X_state[:,i], uk, dt, f) + X_noise[:,i+1]
    X_observed[:,i+1] = h(X_state[:,i+1]) + Y_noise[:,i+1]


print('10 steps')
start = time.time()
for i in range(0,step):
    ekf.estimate(uk)
    ekf.correct(X_observed[:,i+1])
print(str(time.time() - start) + ' sec')

start = time.time()
for i in range(0,step):
    ukf.estimate(uk)
    ukf.correct(X_observed[:,i+1])
print(str(time.time() - start) + ' sec')

start = time.time()
for i in range(0,step):
    ukfr.estimate(uk)
    ukfr.correct(X_observed[:,i+1])
print(str(time.time() - start) + ' sec')


