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


print('非線形カルマンフィルターデモ比較3')
np.random.seed(1024)

def set_kalman_filter_functions(dt):
    '''
     Lotka-Volterra　被食者-捕食者モデル
        パラメータは「Unscented Kalman Filterを用いた力学系の状態 およびパラメータ推定」
        システム制御情報学会論文誌,Vol. 24, No. 9, pp. 231–239, 2011より
    '''
    #A=0.8
    B=1
    #C=1.2
    D=1

    def f(x, u=None):
        return np.mat([\
            [x[2,0]*x[0,0]-B*x[0,0]*x[1,0]],\
            [-x[3,0]*x[1,0]+D*x[0,0]*x[1,0]],\
            [0.0],\
            [0.0]\
        ])

    def fd(x, u=None):
        return np.mat([\
            [x[0,0]+dt*(x[2,0]*x[0,0]-B*x[0,0]*x[1,0])],\
            [x[1,0]+dt*(-x[3,0]*x[1,0]+D*x[0,0]*x[1,0])],\
            [x[2,0]],\
            [x[3,0]]\
        ])

    def h(x, u=None):
        return np.mat([\
            [x[0,0]],\
            [x[1,0]]\
        ])

    def F(x, u=None):
        return np.mat([[1+dt*(x[2,0]-B*x[1,0]), -dt*B*x[0,0], dt*x[0,0], 0], [dt*D*x[1,0], 1+dt*(-x[3,0]+D*x[0,0]), 0, -dt*x[1,0]], [0,0,1,0], [0,0,0,1]])

    def H(x=None, u=None):
        return np.mat([[1,0,0,0],[0,1,0,0]])

    return f, fd, h, F, H

calculate_sec = 50
dt = 0.01
step = int(calculate_sec / dt)
T = np.arange(step+1)

x_dim = 4
y_dim = 2
u_dim = 1

x0 = np.mat([[10.0],[10.0],[0.8],[1.2]])
U = np.mat([range(step+1)])

Q = np.mat([[ 0.0,0.0,0.0,0.0 ],[ 0.0,0.0,0.0,0.0 ],[ 0.0,0.0,0.0,0.0 ],[ 0.0,0.0,0.0,0.0 ]])
R = np.mat([[0.05,0],[0,0.05]])

(f, fd, h, F, H) = set_kalman_filter_functions(dt)

x0_estimate = np.mat([[0.01],[0.01],[0.01],[0.01]])
P0 = np.mat([[1,0,0,0],[0,1,0,0],[0,0,0.1,0],[0,0,0,0.1]])

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

fig, axarr = plt.subplots(5)

axarr[0].plot(np.asarray(X_state)[0,:], 'y-', label='true', linewidth=2)
axarr[0].plot(np.asarray(X_observed)[0,:], 'cx', label='observed', linewidth=1)
axarr[0].plot(np.asarray(ekf.get_estimate_value())[0,:], 'g-', label='extended', linewidth=1)
axarr[0].plot(np.asarray(ukf1.get_estimate_value())[0,:], 'm-', label='unscented', linewidth=1)
axarr[0].plot(np.asarray(ukfr1.get_estimate_value())[0,:], 'b-', label='unscented rk', linewidth=1)
axarr[0].set_title('x0', fontsize=8, loc='left', x=1.02, y=0.85)
axarr[0].set_ylabel('val', fontsize=8)
axarr[0].tick_params(labelsize=6)
axarr[0].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[0].grid(True)

axarr[1].plot(np.asarray(X_state)[1,:], 'y-', label='true', linewidth=2)
axarr[1].plot(np.asarray(ekf.get_estimate_value())[1,:], 'g-', label='extended', linewidth=1)
axarr[1].plot(np.asarray(ukf1.get_estimate_value())[1,:], 'm-', label='unscented', linewidth=1)
axarr[1].plot(np.asarray(ukfr1.get_estimate_value())[1,:], 'b-', label='unscented rk', linewidth=1)
axarr[1].set_title('x1', fontsize=8, loc='left', x=1.02, y=0.85)
axarr[1].set_ylabel('val', fontsize=8)
axarr[1].tick_params(labelsize=6)
axarr[1].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[1].grid(True)

axarr[2].plot(np.asarray(X_state)[2,:], 'y-', label='true', linewidth=2)
axarr[2].plot(np.asarray(ekf.get_estimate_value())[2,:], 'g-', label='extended', linewidth=1)
axarr[2].plot(np.asarray(ukf1.get_estimate_value())[2,:], 'm-', label='unscented', linewidth=1)
axarr[2].plot(np.asarray(ukfr1.get_estimate_value())[2,:], 'b-', label='unscented rk', linewidth=1)
axarr[2].set_title('x2', fontsize=8, loc='left', x=1.02, y=0.85)
axarr[2].set_ylabel('val', fontsize=8)
axarr[2].tick_params(labelsize=6)
axarr[2].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[2].grid(True)

axarr[3].plot(np.asarray(X_state)[3,:], 'y-', label='true', linewidth=2)
axarr[3].plot(np.asarray(ekf.get_estimate_value())[3,:], 'g-', label='extended', linewidth=1)
axarr[3].plot(np.asarray(ukf1.get_estimate_value())[3,:], 'm-', label='unscented', linewidth=1)
axarr[3].plot(np.asarray(ukfr1.get_estimate_value())[3,:], 'b-', label='unscented rk', linewidth=1)
axarr[3].set_title('x3', fontsize=8, loc='left', x=1.02, y=0.85)
axarr[3].set_ylabel('val', fontsize=8)
axarr[3].tick_params(labelsize=6)
axarr[3].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[3].grid(True)

ekf_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ekf.get_estimate_value()))**2).sum(axis=0))
ukf1_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf1.get_estimate_value()))**2).sum(axis=0))
ukf2_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf2.get_estimate_value()))**2).sum(axis=0))
ukf3_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf3.get_estimate_value()))**2).sum(axis=0))
ukfr1_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukfr1.get_estimate_value()))**2).sum(axis=0))
ukfr2_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukfr2.get_estimate_value()))**2).sum(axis=0))

axarr[4].plot(T,ekf_diff, 'r-', label='extended '+' (sum'+str(round(ekf_diff.sum(),2))+')', linewidth=1)
axarr[4].plot(T,ukf1_diff, 'b-', label='unscented '+str(ukf1.kappa)+ ' (sum'+str(round(ukf1_diff.sum(),0))+')', linewidth=1)
axarr[4].plot(T,ukf2_diff, 'g-', label='unscented '+str(ukf2.kappa)+ ' (sum'+str(round(ukf2_diff.sum(),0))+')', linewidth=1)
axarr[4].plot(T,ukf3_diff, 'y-', label='unscented '+str(ukf3.kappa)+ ' (sum'+str(round(ukf3_diff.sum(),0))+')', linewidth=1)
axarr[4].plot(T,ukfr1_diff, 'm-', label='unscented rk'+str(ukfr1.kappa)+ ' (sum'+str(round(ukfr1_diff.sum(),0))+')', linewidth=1)
axarr[4].plot(T,ukfr2_diff, 'c-', label='unscented rk'+str(ukfr2.kappa)+ ' (sum'+str(round(ukfr2_diff.sum(),0))+')', linewidth=1)
axarr[4].set_title('diff from true to estimate', fontsize=8, loc='left', x=1.02, y=0.85)
axarr[4].set_ylabel('diff', fontsize=8)
axarr[4].tick_params(labelsize=6)
axarr[4].legend(fontsize=8, bbox_to_anchor=(1.01, 0.95), loc='upper left')
axarr[4].grid(True)

plt.subplots_adjust(right=0.7)
plt.savefig(os.path.join(os.path.dirname(__file__), 'img/nonlinear_kalman_filter_ex3.png'))