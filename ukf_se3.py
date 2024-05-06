import numpy as np
from se3_math import *
from motion import *

# https://arxiv.org/pdf/2002.00878.pdf
# https://arxiv.org/pdf/1806.11012.pdf

def sigmapoint(s: State, cov, func1, func2=None):
    if func2 is None:
        func2 = func1

    M = s.order
    lambda_ = 1e-3

    w0 = lambda_ / (lambda_ + M)
    w1 = 0.5 / (lambda_ + M)

    L = np.linalg.cholesky((M + lambda_) * cov)

    vec1 = func1(s)
    vec2 = func2(s)

    mean_sst = w0 * np.matmul(vec1[:, np.newaxis], vec2[:, np.newaxis].transpose())

    for i in range(M):
        delta = L[:, i]

        pert_state = s.add(delta)# phi(s, delta)

        # print(pert_state.T_wfr, delta)
        vec1 = func1(pert_state)
        vec2 = func2(pert_state)
        
        mean_sst += w1 * np.matmul(vec1[:, np.newaxis], vec2[:, np.newaxis].transpose())

        pert_state = s.add(-delta) #phi(s, -delta)
        vec1 = func1(pert_state)
        vec2 = func2(pert_state)
        
        mean_sst += w1 * np.matmul(vec1[:, np.newaxis], vec2[:, np.newaxis].transpose())

    return mean_sst


class UKF:
    def __init__(self, model, initial_state, initial_cov, measurement_noise = np.eye(6)*1e-6, system_noise = np.zeros((8, 8))):
        self.model = model
        self.x = initial_state
        self.P = initial_cov # 8x8
        self.R = measurement_noise # 6x6, noise of the odometry twist
        self.Q = system_noise # 8x8

        self.last_measurement_time_s = -1

    def predict(self, delta_T): # List [pred_state, pred_cov]
        new_state = self.model.predict(self.x, delta_T)

        # sigmapoint approx of a covariance for the predicted state

        def F(pert_prev_state):
            pert_new_state = self.model.predict(pert_prev_state, delta_T)

            # print(pert_prev_state.T_wfr, delta_T)
            return pert_new_state.substract(new_state)

        new_cov = sigmapoint(self.x, self.P, F)
        return new_state, new_cov


    def update(self, z_pose_se3, timestamp_s):
        if self.last_measurement_time_s < 0:
            self.last_measurement_time_s = timestamp_s
            self.x.T_wfr = z_pose_se3
            return self.x, self.P

        delta_T = timestamp_s - self.last_measurement_time_s
        self.last_measurement_time_s = timestamp_s
        
        pred_state, P_xx = self.predict(delta_T)
        P_xx = P_xx + self.Q
        
        pred_obs = measurement_function(pred_state) # y^ = h(x^)

        def Fy(pert_pred_state):
            return  measurement_function(pert_pred_state).substract(pred_obs) # Log(y^-1 h(x^exp(ksi)))

        def Fx(pert_pred_state):
            return pert_pred_state.substract(pred_state)

        P_yy = sigmapoint(pred_state, P_xx, Fy) + self.R
        P_xy = sigmapoint(pred_state, P_xx, Fx, Fy)

        K = np.matmul(P_xy, np.linalg.inv(P_yy))

        updated_state = pred_state.add(np.matmul(K, Measurement(z_pose_se3).substract(pred_obs)))
        updated_cov = P_xx - np.matmul(K, np.matmul(P_yy, K.transpose()))

        self.x = updated_state
        self.P = convert_to_positive_semi_definite(updated_cov, 1e-9)
        return self.x, self.P
        



























        
        
            
        