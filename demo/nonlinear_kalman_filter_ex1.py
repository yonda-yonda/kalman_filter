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


print('非線形カルマンフィルターデモ比較1')
np.random.seed(1024)

def set_kalman_filter_functions(dt):
    '''
    「カルマンフィルタの基礎」足立修一,丸田一郎 例題7.4
        状態推定のベンチマーク
    '''
    def f(x, u):
        return np.mat([[ 0.2*x[0,0] + 25*x[0,0]/(1 + x[0,0]*x[0,0]) + 8*math.cos(1.2*u[0,0]) ]])

    def h(x, u=None):
        return np.mat([[ x[0,0]*x[0,0]/20 ]])

    def F(x, u=None):
        return np.mat([[ 0.2 + 25/(1+x[0,0]*x[0,0]) - 2*x[0,0]*x[0,0]/((1 + x[0,0]*x[0,0])*(1 + x[0,0]*x[0,0])) ]])

    def H(x=None, u=None):
        return np.mat([[ x[0,0]/10 ]])

    return f, h, F, H

dt = None
step = 50
T = np.arange(step+1)

x_dim = 1
y_dim = 1
u_dim = 1

x0 = np.mat([[0]])
U = np.mat([range(step+1)])

Q = np.mat([[ 3.0 ]])
R = np.mat([[ 1.0 ]])

(f, h, F, H) = set_kalman_filter_functions(dt)

X_noise = np.mat(np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, step+1)).T
Y_noise = np.mat(np.random.multivariate_normal(np.zeros(R.shape[0]), R, step+1)).T
X_state = np.mat(np.zeros((x_dim,step+1)))
X_state[:,0] = x0 + X_noise[:,0]
X_observed = np.mat(np.zeros((y_dim,step+1)))
X_observed[:,0] = h(X_state[:,0]) + Y_noise[:,0]

x0_estimate = np.mat([[0]])
P0 = np.mat([[ 1.0 ]])

ekf = ExtendedKalmanFilter(f, h, F, H, Q, R, x0_estimate, P0, u_dim, step)
ukf1 = UnscentedKalmanFilter(f, h, Q, R, x0_estimate, P0, u_dim, step, 0)
ukf2 = UnscentedKalmanFilter(f, h, Q, R, x0_estimate, P0, u_dim, step, 0.5)
ukf3 = UnscentedKalmanFilter(f, h, Q, R, x0_estimate, P0, u_dim, step, 1.0)


for i in range(0,step):
    uk = U[:,i]
    X_state[:,i+1] = f(X_state[:,i], uk) + X_noise[:,i+1]
    X_observed[:,i+1] = h(X_state[:,i+1]) + Y_noise[:,i+1]
    ekf.estimate(uk)
    ekf.correct(X_observed[:,i+1])
    ukf1.estimate(uk)
    ukf1.correct(X_observed[:,i+1])
    ukf2.estimate(uk)
    ukf2.correct(X_observed[:,i+1])
    ukf3.estimate(uk)
    ukf3.correct(X_observed[:,i+1])

fig, axarr = plt.subplots(3,2)

axarr[0,0].plot(np.asarray(X_state)[0,:], 'y-', label='true', linewidth=2)
axarr[0,0].plot(np.asarray(X_observed)[0,:], 'cx', label='observed', linewidth=1)
axarr[0,0].plot(np.asarray(ekf.get_estimate_value())[0,:], 'g-', label='extended', linewidth=1)
axarr[0,0].set_title('extended', fontsize=8)
axarr[0,0].set_ylabel('val', fontsize=8)
axarr[0,0].tick_params(labelsize=6)
axarr[0,0].legend(fontsize=8)
axarr[0,0].set_ylim(-40, 90)
axarr[0,0].grid(True)

axarr[1,0].plot(np.asarray(X_state)[0,:], 'y-', label='true', linewidth=2)
axarr[1,0].plot(np.asarray(X_observed)[0,:], 'cx', label='observed', linewidth=1)
axarr[1,0].plot(np.asarray(ukf1.get_estimate_value())[0,:], 'g-', label='unscented '+str(ukf1.kappa), linewidth=1)
axarr[1,0].set_title('unscented', fontsize=8)
axarr[1,0].set_ylabel('val', fontsize=8)
axarr[1,0].tick_params(labelsize=6)
axarr[1,0].legend(fontsize=8)
axarr[1,0].set_ylim(-40, 90)
axarr[1,0].grid(True)

ekf_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ekf.get_estimate_value()))**2).sum(axis=0))
ukf1_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf1.get_estimate_value()))**2).sum(axis=0))

axarr[2,0].plot(T,ekf_diff, 'r-', label='extended '+' (sum'+str(round(ekf_diff.sum(),2))+')', linewidth=1)
axarr[2,0].plot(T,ukf1_diff, 'b-', label='unscented '+str(ukf1.kappa)+ ' (sum'+str(round(ukf1_diff.sum(),2))+')', linewidth=1)
axarr[2,0].set_title('diff from true to estimate', fontsize=8)
axarr[2,0].set_ylabel('diff', fontsize=8)
axarr[2,0].tick_params(labelsize=6)
axarr[2,0].legend(fontsize=8)
axarr[2,0].set_ylim(0, 90)
axarr[2,0].grid(True)

axarr[0,1].plot(np.asarray(X_state)[0,:], 'y-', label='true', linewidth=2)
axarr[0,1].plot(np.asarray(X_observed)[0,:], 'cx', label='observed', linewidth=1)
axarr[0,1].plot(np.asarray(ukf2.get_estimate_value())[0,:], 'g-', label='unscented '+str(ukf2.kappa), linewidth=1)
axarr[0,1].set_title('extended', fontsize=8)
axarr[0,1].set_ylabel('val', fontsize=8)
axarr[0,1].tick_params(labelsize=6)
axarr[0,1].legend(fontsize=8)
axarr[0,1].set_ylim(-40, 90)
axarr[0,1].grid(True)

axarr[1,1].plot(np.asarray(X_state)[0,:], 'y-', label='true', linewidth=2)
axarr[1,1].plot(np.asarray(X_observed)[0,:], 'cx', label='observed', linewidth=1)
axarr[1,1].plot(np.asarray(ukf3.get_estimate_value())[0,:], 'g-',label='unscented '+str(ukf3.kappa), linewidth=1)
axarr[1,1].set_title('unscented', fontsize=8)
axarr[1,1].set_ylabel('val', fontsize=8)
axarr[1,1].tick_params(labelsize=6)
axarr[1,1].legend(fontsize=8)
axarr[1,1].set_ylim(-40, 90)
axarr[1,1].grid(True)

ukf2_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf2.get_estimate_value()))**2).sum(axis=0))
ukf3_diff = np.sqrt((np.asarray(np.asarray(X_state)-np.asarray(ukf3.get_estimate_value()))**2).sum(axis=0))

axarr[2,1].plot(T,ukf2_diff, 'r-', label='unscented '+str(ukf2.kappa)+ ' (sum'+str(round(ukf2_diff.sum(),2))+')', linewidth=1)
axarr[2,1].plot(T,ukf3_diff, 'b-', label='unscented '+str(ukf3.kappa)+ ' (sum'+str(round(ukf3_diff.sum(),2))+')', linewidth=1)
axarr[2,1].set_title('diff from true to estimate', fontsize=8)
axarr[2,1].set_ylabel('diff', fontsize=8)
axarr[2,1].tick_params(labelsize=6)
axarr[2,1].legend(fontsize=8)
axarr[2,1].set_ylim(0, 90)
axarr[2,1].grid(True)

plt.savefig(os.path.join(os.path.dirname(__file__), 'img/nonlinear_kalman_filter_ex1.png'))