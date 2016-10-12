#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import math
import numpy as np

class UnscentedKalmanFilter(object):
    """
        無香料カルマンフィルター
        初期値、制御量、観測量から線形システムの真の状態量を推定する。
        状態量: Xk+1 = f(Xk, Uk) + v
        観測量: Yk+1 = h(Xk+1) + w
        推定量は X(t_dim, x_dim)の numpy.matrix で保持する
        マルチレート処理ができるよう予測と更新のメソッドを分けている。
    """
    def __init__(self, f, h, Q, R, x0_estimate, P0, u_dim, step, kappa=0.0):
        """
            Args:
            f : f(xt, ut)、返り値はx0の次元と一致するnumpy.ndarray 1元配列
            h : h(xt)、返り値はy_dimと一致するnumpy.ndarray 1元配列
            F : df(xt, ut)/dx、返り値はx_dim * x_dimのnumpy.ndarray 2元配列
            H : df(xt, ut)/dx、返り値はy_dim * x_dimのnumpy.ndarray 2元配列
            Q : 状態量方程式の誤差共分散行列 numpy.ndarray 2元配列
            R : 観測量方程式の誤差共分散行列 umpy.ndarray 2元配列
            x0_estimate : 状態量初期値 numpy.ndarray 1次元配列
            P0 : 誤差共分散行列の初期値 numpy.ndarray 2元配列
            u_dim : int 制御量の次元
            step : int 最大回数
            kappa : 推定パラメータ
        """
        # pythonは参照渡し、arrayは可変性オブジェクトなため元の配列を変更しないようcopy
        # 初期値設定
        assert isinstance(Q, np.matrixlib.defmatrix.matrix) and \
                Q.shape[0] == Q.shape[1], \
               'Qはnumpyのmatrixかつ正方行列である必要があります。'
        self.Q = Q.copy()
        self.x_dim = Q.shape[0] # 状態量次元
        assert isinstance(R, np.matrixlib.defmatrix.matrix) and \
                R.shape[0] == R.shape[1], \
               'Rはnumpyのmatrixかつ正方行列である必要があります。'
        self.R = R.copy()
        self.y_dim = R.shape[0]

        self.u_dim = u_dim # 制御量次元

        # 関数チェック
        self.check_system_function(f, h)

        assert isinstance(x0_estimate, np.matrixlib.defmatrix.matrix) and x0_estimate.shape[1] == 1, \
               'x0はnumpyのmatrixである必要があります。'

        assert isinstance(P0, np.matrixlib.defmatrix.matrix) and P0.shape[0] == P0.shape[1] and \
               P0.shape[0] == self.x_dim, \
               'P0は正方行列である必要があります。' \
               'またP0はx_dim * x_dimのnumpyのmatrixである必要があります。'
        self.Pk = P0.copy()

        self.t_dim = step + 1 # 時刻次元
        self.kappa = kappa
        self.__k = 0
        self.__k_correct=0
        self.X = np.mat(np.zeros((self.x_dim, self.t_dim)))
        self.X[:,self.__k] = x0_estimate.copy()

        self.omega = 0.5 / (self.x_dim+self.kappa)
        self.omega0 = self.kappa / (self.x_dim+self.kappa)


    def check_system_function(self, f, h):
        """
            関数が正しいかチェック
        """
        assert isinstance(f, types.FunctionType), \
               'fは関数である必要があります。'

        x = np.mat(np.random.randn(self.x_dim,1))
        u = np.mat(np.random.randn(self.u_dim,1))
        x_next = f(x, u)

        assert isinstance(x_next, np.matrixlib.defmatrix.matrix) , \
               'fの返り値はnumpyのmatrixである必要があります。'
        assert x_next.shape[0] == self.x_dim and x_next.shape[1] == 1, \
               'fの返り値は長さx_dimの1次元配列である必要があります。'
        self.f = f

        assert isinstance(h, types.FunctionType), \
               'hは関数である必要があります。'
        y = h(x)
        assert isinstance(y, np.matrixlib.defmatrix.matrix) , \
               'hの返り値はnumpyのmatrixである必要があります。'
        assert y.shape[0] == self.y_dim and y.shape[1] == 1, \
               'hの返り値は長さy_dimの1次元配列である必要があります。'
        self.h = h


    def estimate(self, u):
        """
            Args:
            u : numpy.matrix 長さu_dimの1次元配列

            Return:
            boolean
            更新ができた場合はTrue、更新回数が与えられたステップ数を越えた場合はFalse

            制御量とstep k-1の状態量からstep kの値を予測する。
            更新回数が与えられたステップ数を越えた場合は更新しない。
        """
        if self.__k + 1 >= self.t_dim:
            return False

        assert isinstance(u, np.matrixlib.defmatrix.matrix) and u.shape[0] == self.u_dim and u.shape[1] == 1, \
               'uは長さu_dimのnumpyのmatrixである必要があります。'

        X = self.get_current_estimate_value()

        X_sigma = np.mat(np.zeros((self.x_dim,1 + self.x_dim * 2)))
        X_sigma[:,0] = X
        P_cholesky =  np.linalg.cholesky(self.Pk)
        for i in range(self.x_dim):
            diff = math.sqrt(self.x_dim + self.kappa) * P_cholesky[:,i]
            X_sigma[:,i+1] = X  + diff
            X_sigma[:,self.x_dim+i+1] = X  - diff

        for i in range(2*self.x_dim+1):
            X_sigma[:,i] = self.f(X_sigma[:,i],u)

        Xk_priori =  self.omega0 * X_sigma[:,0]
        for i in range(1,2*self.x_dim+1):
            Xk_priori += self.omega * X_sigma[:,i]

        diff = X_sigma[:,0] - Xk_priori
        P_priori = self.Q + self.omega0 * diff * diff.T
        for i in range(1,2*self.x_dim+1):
            diff = X_sigma[:,i] - Xk_priori
            P_priori += self.omega * diff * diff.T

        self.__k += 1
        self.X[:,self.__k] = Xk_priori
        self.Pk = P_priori

    def correct(self, Y):
        """
            Args:
            Y : numpy.matrix 長さy_dimの1次元配列

            Return:
            boolean
            補正ができた場合はTrue、補正しなかった場合False

            観測量とstep kの推定量からstep kの値を更新する。
            前回更新したstepを __k_correct として記憶しており、
            step kがこれよりも大きくない（予測されていない）場合は更新しない。
        """
        if self.__k_correct >= self.__k:
            return False

        assert isinstance(Y, np.matrixlib.defmatrix.matrix) and Y.shape[0] == self.y_dim and Y.shape[1] == 1, \
               'Yは長さy_dimのnumpyのmatrixである必要があります。'

        X = self.get_current_estimate_value()

        X_sigma = np.mat(np.zeros((self.x_dim,1 + self.x_dim * 2)))
        X_sigma[:,0] = X
        P_cholesky =  np.linalg.cholesky(self.Pk)
        for i in range(self.x_dim):
            diff = math.sqrt(self.x_dim + self.kappa) * P_cholesky[:,i]
            X_sigma[:,i+1] = X  + diff
            X_sigma[:,self.x_dim+i+1] = X  - diff

        Y_sigma = np.mat(np.zeros((self.y_dim,1 + self.x_dim * 2)))

        for i in range(2*self.x_dim+1):
            Y_sigma[:,i] = self.h(X_sigma[:,i])

        Y_estimate =  self.omega0 * Y_sigma[:,0]
        for i in range(1,2*self.x_dim+1):
            Y_estimate += self.omega * Y_sigma[:,i]

        diff = Y_sigma[:,0] - Y_estimate
        P_yy = self.R + self.omega0 * diff*diff.T
        for i in range(1,2*self.x_dim+1):
            diff = Y_sigma[:,i] - Y_estimate
            P_yy += self.omega * diff*diff.T

        xdiff = X_sigma[:,0] - X
        ydiff = Y_sigma[:,0] - Y_estimate
        P_xy = self.omega0 * xdiff*ydiff.T
        for i in range(1,2*self.x_dim+1):
            xdiff = X_sigma[:,i] - X
            ydiff = Y_sigma[:,i] - Y_estimate
            P_xy += self.omega * xdiff * ydiff.T

        K_gain = P_xy * np.linalg.inv(P_yy)
        self.X[:,self.__k] = X + K_gain * (Y-Y_estimate)
        self.Pk = self.Pk - K_gain * P_xy.T

        self.__k_correct = self.__k

        return True

    def get_estimate_value(self):
        """
            Return:
            numpy.matrix
            状態量（ステップ0から最新ステップ）
        """
        return self.X[:,:self.__k+1]

    def get_current_estimate_value(self):
        """
            Return:
            numpy.matrix
            状態量（最新ステップ）
        """
        return self.X[:,self.__k]


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


    print('無香料カルマンフィルターデモ')
    print('状態量とともに一部のパラメータも推定する。')
    np.random.seed(512)


    def f(x, u):
        return np.mat([[x[0,0]*x[1,0] + u[0,0]*x[2,0]], [x[1,0]], [x[2,0]]])

    def h(x, u=None):
        return np.mat([x[0,0]])


    step = 1000
    T = np.arange(step+1)

    x_dim = 3
    y_dim = 1
    u_dim = 1

    x0 = np.mat('0; 0.9; 1')
    U = np.mat(0.05*np.sin(0.01 * np.pi * T))

    Q = np.mat([[0.36,0,0],[0,0,0],[0,0,0]])
    R = np.mat(1.6)

    X_noise = np.mat(np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, step+1)).T
    Y_noise = np.mat(np.random.multivariate_normal(np.zeros(R.shape[0]), R, step+1)).T
    X_state = np.mat(np.zeros((x_dim,step+1)))
    X_state[:,0] = x0 + X_noise[:,0]
    X_observed = np.mat(np.zeros((y_dim,step+1)))
    X_observed[:,0] = h(X_state[:,0]) + Y_noise[:,0]

    x0_estimate = np.mat('0; 0; 0')
    P0 = np.mat([[4,0,0],[0,4,0],[0,0,4]])

    def make_figures(X_state, X_observed, kalman_filter_object):
        fig, axarr = plt.subplots(2)

        axarr[0].plot(T,np.asarray(X_state)[0,:], 'b-', linewidth=1)
        axarr[0].plot(T,np.asarray(X_observed)[0,:], 'cx', linewidth=1)
        axarr[0].plot(T,np.asarray(kalman_filter_object.get_estimate_value())[0,:], 'r-', linewidth=1)
        axarr[0].set_title('state value', fontsize=8)
        axarr[0].set_ylabel('val', fontsize=8)
        axarr[0].tick_params(labelsize=6)
        axarr[0].set_ylim(-6, 6)
        axarr[0].grid(True)

        axarr[1].plot(T,np.asarray(X_state)[1,:], 'r-', linewidth=1)
        axarr[1].plot(T,np.asarray(kalman_filter_object.get_estimate_value())[1,:], 'r-', linewidth=1)
        axarr[1].plot(T,np.asarray(X_state)[2,:], 'b-', linewidth=1)
        axarr[1].plot(T,np.asarray(kalman_filter_object.get_estimate_value())[2,:], 'b-', linewidth=1)
        axarr[1].set_title('constant value', fontsize=8)
        axarr[1].set_ylabel('val', fontsize=8)
        axarr[1].tick_params(labelsize=6)
        axarr[1].set_ylim(-1, 2)
        axarr[1].grid(True)

        plt.savefig('demo/img/unscented_kalman_filter_'+ str(kalman_filter_object.kappa).replace(".", "_") +'.png')


    kf1 = UnscentedKalmanFilter(f, h, Q, R, x0_estimate, P0, u_dim, step, 0.5)
    kf2 = UnscentedKalmanFilter(f, h, Q, R, x0_estimate, P0, u_dim, step, 1.0)
    kf3 = UnscentedKalmanFilter(f, h, Q, R, x0_estimate, P0, u_dim, step, 5.0)

    for i in range(0,step):
        uk = U[:,i]
        X_state[:,i+1] = f(X_state[:,i], uk) + X_noise[:,i+1]
        X_observed[:,i+1] = h(X_state[:,i+1]) + Y_noise[:,i+1]
        kf1.estimate(uk)
        kf1.correct(X_observed[:,i+1])
        kf2.estimate(uk)
        kf2.correct(X_observed[:,i+1])
        kf3.estimate(uk)
        kf3.correct(X_observed[:,i+1])

    make_figures(X_state, X_observed, kf1)
    make_figures(X_state, X_observed, kf2)
    make_figures(X_state, X_observed, kf3)


    kf0 = UnscentedKalmanFilter(f, h, Q, R, x0_estimate, P0, u_dim, step)
    print('更新により予測だけの値よりも真値に近づいている。')

    for i in range(100):
        uk = U[:,i]
        kf0.estimate(uk)
        kf0.correct(X_observed[:,i+1])
    for j in range(10):
        i = j + 100
        uk = U[:,i]
        kf0.estimate(uk)
        print('step', i+1,' 真値:')
        print(X_state[:,i+1].T)
        print('step', i+1,' 予測:')
        print(kf0.get_current_estimate_value().T)
        kf0.correct(X_observed[:,i+1].T)
        print('step', i+1,' 更新:')
        print(kf0.get_current_estimate_value().T)


    print('更新前 ')
    print(kf0.get_current_estimate_value().T)
    kf0.correct(np.mat("100;100"))
    print('予測なしで更新 ')
    print(kf0.get_current_estimate_value().T)
    print('estimateを呼ばずにcorrectを呼んでも値は更新されない。')
