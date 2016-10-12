#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class KalmanFilter(object):
    """
        カルマンフィルター
        初期値、制御量、観測量から線形システムの真の状態量を推定する。
        状態量: Xk+1 = A * Xk + B * uk + v
        観測量: Yk+1 = H * Xk+1 + w
        推定量は numpy.matrix で保持する。
        マルチレート処理ができるよう予測と更新のメソッドを分けている。
    """
    def __init__(self, A, B, H, Q, R, P0, x0, step):
        """
            Args:
            A, B, H, Q, R, P0 : numpy.matrix 2次元配列
            x0 : numpy.matrix の1次元配列

            すべて numpy.matrix で保持する。
        """
        # pythonは参照渡し、matrixは可変性オブジェクトなため元の配列を変更しないようcopy

        assert isinstance(A, np.matrixlib.defmatrix.matrix) and A.shape[0] == A.shape[1], \
               'Aはnumpyのmatrixかつ正方行列である必要があります。'
        self.A = A.copy()
        self.x_dim = A.shape[1]

        assert isinstance(B, np.matrixlib.defmatrix.matrix) and B.shape[0] == A.shape[0], \
               'BはnumpyのmatrixかつAの行数と一致する必要があります。'
        self.B = B.copy()
        self.u_dim = B.shape[1]

        assert isinstance(H, np.matrixlib.defmatrix.matrix) and H.shape[1] == A.shape[1], \
               'HはnumpyのmatrixかつAの列数と一致する必要があります。'
        self.H = H.copy()
        self.y_dim = H.shape[0]

        assert isinstance(Q, np.matrixlib.defmatrix.matrix) and \
                Q.shape[0] == Q.shape[1] and Q.shape[0] == A.shape[0], \
               'Qはnumpyのmatrixかつ正方行列である必要があります。' \
               'またQとAの次元が一致する必要があります。'
        self.Q = Q.copy()
        assert isinstance(R, np.matrixlib.defmatrix.matrix) and \
                R.shape[0] == R.shape[1]and R.shape[0] == H.shape[0], \
               'Rはnumpyのmatrixかつ正方行列である必要があります。' \
               'またRとHの行数が一致する必要があります。'
        self.R = R.copy()

        self.t_dim = step + 1 # 時刻次元

        # 初期値設定
        self.setup(x0, P0)


    def setup(self, x0, P0):
        """
            Args:
            x0 : numpy.matrix の1次元配列
            P0 : numpy.matrix

            状態量と誤差の共分散行列の初期値を設定する。
        """
        assert isinstance(x0, np.matrixlib.defmatrix.matrix) and x0.shape[0] == self.x_dim and x0.shape[1] == 1, \
            'x0はnumpyのmatrixかつ1次元の配列かつ係数行列で定義される状態量の長さと一致する必要があります。'
        X0 = x0.copy()

        assert isinstance(P0, np.matrixlib.defmatrix.matrix) and P0.shape[0] == P0.shape[1]and \
               P0.shape[0] == self.A.shape[0], \
               'Pはnumpyのmatrixかつ正方行列である必要があります。' \
               'またPとAの行数が一致する必要があります。'
        self.Pk = P0.copy()

        self.__k = 0
        self.X = np.mat(np.zeros((self.x_dim, self.t_dim)))
        self.X[:,self.__k ] = X0
        self.__k_correct = 0


    def estimate(self, u):
        """
            予測
            Args:
            u : numpy.matrix の1次元配列

            Return:
            boolean
            更新ができた場合はTrue、更新回数が与えられたステップ数を越えた場合はFalse

            制御量とstep k-1の状態量からstep kの値を予測する。
            更新回数が与えられたステップ数を越えた場合は更新しない。
        """
        if self.__k + 1 >= self.t_dim:
            return False


        assert isinstance(u, np.matrixlib.defmatrix.matrix) and u.shape[0] == self.u_dim and u.shape[1] == 1, \
            'uはnumpyのmatrixかつ1次元の配列かつ係数行列で定義される制御量の長さと一致する必要があります。'
        Uk = u.copy()

        self.__k += 1
        Xk_priori = self.A * self.X[:,self.__k-1] + self.B * Uk
        P_priori = self.A * self.Pk * self.A.T + self.Q

        self.X[:,self.__k] = Xk_priori
        self.Pk = P_priori

    def correct(self, x_observed):
        """
            更新
            Args:
            x_observed : numpy.ndarray または numpy.matrix の1次元配列

            Return:
            boolean
            補正ができた場合はTrue、補正しなかった場合False

            観測量とstep kの推定量からstep kの値を更新する。
            前回更新したstepを __k_correct として記憶しており、
            step kがこれよりも大きくない（予測されていない）場合は更新しない。
        """
        if self.__k_correct >= self.__k:
            return False

        assert isinstance(x_observed, np.matrixlib.defmatrix.matrix) and x_observed.shape[0] == self.x_dim and x_observed.shape[1] == 1, \
            'x_observedはnumpyのmatrixかつ1次元の配列かつ係数行列で定義される状態量の長さと一致する必要があります。'
        X_observed = x_observed.copy()

        Xk = self.get_current_estimate_value()

        S = self.R + self.H * self.Pk * self.H.T
        K_gain = self.Pk * self.H.T * S.I
        Xk_posteriori = Xk + K_gain * (X_observed - self.H * Xk)
        P_posteriori = (np.mat(np.identity(self.x_dim)) - K_gain * self.H) * self.Pk

        self.X[:,self.__k] = Xk_posteriori
        self.Pk = P_posteriori
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


    print('カルマンフィルターデモ')
    np.random.seed(512)

    step = 100
    A = np.mat([[1,0],[0,1]])
    B = np.mat([[1],[1]])
    H = np.mat([[1,0],[0,1]])
    Q = np.mat([[0,0],[0,0]])
    R = np.mat([[2,0],[0,0.02]])
    x0 = np.mat([[2.5],[-1.2]])
    U = np.mat(np.zeros((1,step)))

    X_noise = np.mat(np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, step+1)).T
    Y_noise = np.mat(np.random.multivariate_normal(np.zeros(R.shape[0]), R, step+1)).T
    X_state = np.mat(np.zeros((2,step+1)))

    X_state[:,0] = x0 + X_noise[:,0]
    X_observed = np.mat(np.zeros((2,step+1)))
    X_observed[:,0] = H * X_state[:,0] + Y_noise[:,0]

    P0 = np.mat([[1,0],[0,1]])
    x0_estimate = np.mat([[0],[0]])
    kf = KalmanFilter(A,B,H,Q,R,P0,x0_estimate,step)

    for i in range(0,step):
        Uk = U[:,i]
        X_state[:,i+1] = A * X_state[:,i] + B * Uk + X_noise[:,i+1]
        X_observed[:,i+1] = H * X_state[:,i+1] + Y_noise[:,i+1]

        kf.estimate(Uk)
        kf.correct(X_observed[:,i+1])

    plt.figure()
    plt.plot(np.asarray(X_state)[0,:], 'b-', label='x1:true', linewidth=2)
    plt.plot(np.asarray(X_observed)[0,:], 'b:x', label='x1:noise', linewidth=1)
    plt.plot(np.asarray(kf.get_estimate_value())[0,:], 'b-', label='x1:estimated', linewidth=1)
    plt.plot(np.asarray(X_state)[1,:], 'r-', label='x2:true', linewidth=2)
    plt.plot(np.asarray(X_observed)[1,:], 'r:x', label='x2:noise', linewidth=1)
    plt.plot(np.asarray(kf.get_estimate_value())[1,:], 'r-', label='x2:estimated', linewidth=1)
    plt.xlabel('step')
    plt.ylabel('val')
    plt.legend()
    plt.savefig('demo/img/kalman_filter_constant_value.png')


    kf2 = KalmanFilter(A,B,H,Q,R,P0,x0_estimate,step)
    print('安定してからは更新により予測だけの値よりも真値に近づいている。')

    for i in range(0,step):
        Uk = U[:,i]
        kf2.estimate(Uk)
        if i>90:
            print('step', i,' 真値:', np.asarray(X_state)[:,i].T )
            print('step', i,' 予測:', kf2.get_current_estimate_value().T)
        kf2.correct(X_observed[:,i+1])
        if i>90:
            print('step', i,' 更新:', kf2.get_current_estimate_value().T)

    print('更新前 ', kf2.get_current_estimate_value().T)
    kf2.correct(np.asarray(X_observed)[:,step-1])
    print('予測なしで更新 ', kf2.get_current_estimate_value().T)
    print('estimateを呼ばずにcorrectを呼んでも値は更新されない。')