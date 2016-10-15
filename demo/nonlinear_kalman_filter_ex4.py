#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import numpy as np
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from extended_kalman_filter import *
from unscented_kalman_filter import *
from unscented_kalman_filter_rungekutta import *


print('非線形カルマンフィルターデモ比較4')
np.random.seed(1024)

def set_kalman_filter_functions(dt):
    '''
     周回運動モデル
        パラメータは「Unscented Kalman Filterを用いた力学系の状態 およびパラメータ推定」
        システム制御情報学会論文誌,Vol. 24, No. 9, pp. 231–239, 2011より
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

calculate_sec = 50
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
ukf1 = UnscentedKalmanFilter(fd, h, Q, R, x0_estimate, P0, u_dim, step, 0)
ukf2 = UnscentedKalmanFilter(fd, h, Q, R, x0_estimate, P0, u_dim, step, 0.5)
ukf3 = UnscentedKalmanFilter(fd, h, Q, R, x0_estimate, P0, u_dim, step, 1.0)
ukfr1 = UnscentedKalmanFilterRungeKutta(f, h, Q, R, x0_estimate, P0, u_dim, step, dt, 0)
ukfr2 = UnscentedKalmanFilterRungeKutta(f, h, Q, R, x0_estimate, P0, u_dim, step, dt, 1.0)


for i in range(0,step):
    uk = U[:,i]
    X_state[:,i+1] = UnscentedKalmanFilterRungeKutta.runge_kutta(X_state[:,i], uk, dt, f) + X_noise[:,i+1]
    X_observed[:,i+1] = h(X_state[:,i+1]) + Y_noise[:,i+1]
    ekf.estimate(uk)
    ekf.correct(X_observed[:,i+1])
    ukf1.estimate(uk)
    ukf1.correct(X_observed[:,i+1])
    ukf2.estimate(uk)
    ukf2.correct(X_observed[:,i+1])
    ukf3.estimate(uk)
    ukf3.correct(X_observed[:,i+1])
    ukfr1.estimate(uk)
    ukfr1.correct(X_observed[:,i+1])
    ukfr2.estimate(uk)
    ukfr2.correct(X_observed[:,i+1])

fig, axarr = plt.subplots(4)

axarr[0].plot(np.asarray(X_state)[0,:],np.asarray(X_state)[2,:], 'y-', label='true', linewidth=2)
axarr[0].plot(np.asarray(ekf.get_estimate_value())[0,:],np.asarray(ekf.get_estimate_value())[2,:], 'g-', label='extended', linewidth=1)
axarr[0].plot(np.asarray(ukf1.get_estimate_value())[0,:],np.asarray(ukf1.get_estimate_value())[2,:], 'm-', label='unscented', linewidth=1)
axarr[0].plot(np.asarray(ukfr1.get_estimate_value())[0,:],np.asarray(ukfr1.get_estimate_value())[2,:], 'b-', label='unscented rk', linewidth=1)
axarr[0].set_title('position x0 - x2', fontsize=8, loc='left', x=1.02, y=0.88)
axarr[0].tick_params(labelsize=6)
axarr[0].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[0].grid(True)

axarr[1].plot(np.asarray(X_state)[4,:], 'y-', label='true', linewidth=2)
axarr[1].plot(np.asarray(ekf.get_estimate_value())[4,:], 'g-', label='extended', linewidth=1)
axarr[1].plot(np.asarray(ukf1.get_estimate_value())[4,:], 'm-', label='unscented', linewidth=1)
axarr[1].plot(np.asarray(ukfr1.get_estimate_value())[4,:], 'b-', label='unscented rk', linewidth=1)
axarr[1].set_title('x4', fontsize=8, loc='left', x=1.02, y=0.85)
axarr[1].set_ylabel('val', fontsize=8)
axarr[1].tick_params(labelsize=6)
axarr[1].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[1].grid(True)

axarr[2].plot(np.asarray(X_state)[5,:], 'y-', label='true', linewidth=2)
axarr[2].plot(np.asarray(ekf.get_estimate_value())[5,:], 'g-', label='extended', linewidth=1)
axarr[2].plot(np.asarray(ukf1.get_estimate_value())[5,:], 'm-', label='unscented', linewidth=1)
axarr[2].plot(np.asarray(ukfr1.get_estimate_value())[5,:], 'b-', label='unscented rk', linewidth=1)
axarr[2].set_title('x5', fontsize=8, loc='left', x=1.02, y=0.88)
axarr[2].set_ylabel('val', fontsize=8)
axarr[2].tick_params(labelsize=6)
axarr[2].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[2].grid(True)

ekf_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ekf.get_estimate_value()))**2).sum(axis=0))
ukf1_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf1.get_estimate_value()))**2).sum(axis=0))
ukf2_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf2.get_estimate_value()))**2).sum(axis=0))
ukf3_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf3.get_estimate_value()))**2).sum(axis=0))
ukfr1_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukfr1.get_estimate_value()))**2).sum(axis=0))
ukfr2_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukfr2.get_estimate_value()))**2).sum(axis=0))

axarr[3].plot(T,ekf_diff, 'r-', label='extended '+' (sum'+str(round(ekf_diff.sum(),2))+')', linewidth=1)
axarr[3].plot(T,ukf1_diff, 'b-', label='unscented '+str(ukf1.kappa)+ ' (sum'+str(round(ukf1_diff.sum(),0))+')', linewidth=1)
axarr[3].plot(T,ukf2_diff, 'g-', label='unscented '+str(ukf2.kappa)+ ' (sum'+str(round(ukf2_diff.sum(),0))+')', linewidth=1)
axarr[3].plot(T,ukf3_diff, 'y-', label='unscented '+str(ukf3.kappa)+ ' (sum'+str(round(ukf3_diff.sum(),0))+')', linewidth=1)
axarr[3].plot(T,ukfr1_diff, 'm-', label='unscented rk'+str(ukfr1.kappa)+ ' (sum'+str(round(ukfr1_diff.sum(),0))+')', linewidth=1)
axarr[3].plot(T,ukfr2_diff, 'c-', label='unscented rk'+str(ukfr2.kappa)+ ' (sum'+str(round(ukfr2_diff.sum(),0))+')', linewidth=1)
axarr[3].set_title('diff from true to estimate', fontsize=8, loc='left', x=1.02, y=0.88)
axarr[3].set_ylabel('diff', fontsize=8)
axarr[3].tick_params(labelsize=6)
axarr[3].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[3].grid(True)

plt.subplots_adjust(right=0.7)
plt.savefig(os.path.join(os.path.dirname(__file__), 'img/nonlinear_kalman_filter_ex4.png'))