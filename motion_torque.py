import numpy as np
from collections import namedtuple

from se3_math import *

class State:
    def __init__(self, T_wfr, W = np.zeros(2), tau = np.zeros(2)):
        self.T_wfr = T_wfr
        self.W = W # np.array([wl, wr])
        self.tau = tau
        self.order = 10  # static

    def add(self, delta):
        delta_pose = Exp6(delta[:6])
        delta_W = delta[6:8]
        delta_tau = delta[8:]
        return State(np.matmul(self.T_wfr, delta_pose), self.W + delta_W, self.tau + delta_tau)

    def substract(self, s_source):
        return np.hstack([
            Log6(np.matmul(inverse_pose(s_source.T_wfr), self.T_wfr)),
            self.W - s_source.W,
            self.tau - s_source.tau
        ])

class Measurement:
    def __init__(self, T_wfr):
        self.T_wfr = T_wfr
        self.order = 6  # static

    def substract(self, m):
        return Log6(np.matmul(inverse_pose(m.T_wfr), self.T_wfr))

def measurement_function(s:State):  # y = h(x)
    return Measurement(s.T_wfr)

class WheeledRobotMotionModel:
    def __init__(self, T_left_wheel_from_rig, wheel_base: float, wheel_radius: float):
        self.T_rfl = inverse_pose(T_left_wheel_from_rig)
        self.d = wheel_base
        self.Rw = wheel_radius

        self.R_rfl = self.T_rfl[:3,:3]

    def predict(self, s: State, delta_t):
        V = s.W * self.Rw # [vl, vr]

        delta_v = V[1] - V[0]  # vr - vl
        if abs(delta_v) < 1e-6:
            vl = np.array([0, 0, V[0]])
            F = np.hstack([np.zeros(3), np.matmul(self.R_rfl, vl)])

            Delta = np.zeros(s.order)
            Delta[:6] = F * delta_t
            Delta[6:8] = s.tau * delta_t
            return s.add(Delta)

        wl = np.array([0, -delta_v/self.d, 0])
        Pl = np.array([-V[0] * self.d/delta_v, 0, 0])

        Pr = -(np.matmul(self.R_rfl, Pl) + self.T_rfl[:3, 3])
        wr = np.matmul(self.R_rfl, wl)

        Vrig = np.cross(wr, Pr)

        F = np.hstack([wr, Vrig])

        Delta = np.zeros(s.order)
        Delta[:6] = F * delta_t
        Delta[6:8] = s.tau * delta_t
        return s.add(Delta)
        
        
        