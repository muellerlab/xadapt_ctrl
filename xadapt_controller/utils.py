#!/usr/bin/python3

import numpy as np
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
from enum import Enum

import os


class ModelType(Enum):
    BASE_MODEL = 1
    ADAP_MODULE = 2

# TODO: ONNX model conversion, check validity
# TODO: Unit Test


class Model:
    def __init__(self, base_model_path='/benchmark/base_model.onnx', adap_module_path='/benchmark/adap_module.onnx', model_rms_path='/benchmark/base_model.npz'):
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.base_model_path = current_path+base_model_path
        self.adap_module_path = current_path+adap_module_path
        self.model_rms_path = current_path+model_rms_path

        rms_data = np.load(self.model_rms_path)
        self.obs_mean = np.mean(rms_data["mean"], axis=0)
        self.obs_var = np.mean(rms_data["var"], axis=0)

        self.act_mean = np.array([1.0 / 2, 1.0 / 2,
                                  1.0 / 2, 1.0 / 2])[np.newaxis, :]
        self.act_std = np.array([1.0 / 2, 1.0 / 2,
                                 1.0 / 2, 1.0 / 2])[np.newaxis, :]
        self.act_size = 4
        self.state_obs_size = 17
        self.history_len = 400

        self.base_session = None
        self.adap_session = None
        self.base_obs_name = None
        self.adap_obs_name = None

    def set_act_size(self, act_size):
        self.act_size = act_size

    def set_state_obs_size(self, state_obs_size):
        self.state_obs_size = state_obs_size

    def set_history_len(self, history_len):
        self.history_len = history_len
        
    def set_const_sizes(self,state_obs_size,act_size,history_len):
        self.set_act_size(act_size)
        self.set_state_obs_size(state_obs_size)
        self.set_history_len(history_len)

    def activate(self):
        self.base_session = onnxruntime.InferenceSession(
            self.base_model_path, None)
        self.base_obs_name = self.base_session.get_inputs()[0].name
        self.adap_session = onnxruntime.InferenceSession(
            self.adap_module_path, None)
        self.adap_obs_name = self.adap_session.get_inputs()[0].name

    def normalize_obs(self, obs, model_type):
        if model_type is ModelType.BASE_MODEL:
            return (obs - self.obs_mean[:self.state_obs_size+self.act_size]) / np.sqrt(self.obs_var[:self.state_obs_size+self.act_size] + 1e-8)
        else:
            # Normalize for Adaptation module observations
            obs_n_norm = obs.reshape([1, -1])

            # obs_current_n_normalized = obs_n_norm[:,
            #                               :-history_len*(act_size+state_obs_size)]
            # obs_current_normalized = (obs_current_n_normalized - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            # obs_n_norm[:, :-history_len *
            #    (act_size+state_obs_size)] = obs_current_normalized

            obs_state_history_n_normalized = obs_n_norm[:, -self.history_len*(
                self.act_size+self.state_obs_size):-self.history_len*self.act_size]

            obs_state_mean = np.tile(self.obs_mean[:self.state_obs_size], [
                                     1, self.history_len])
            obs_state_var = np.tile(self.obs_var[:self.state_obs_size], [
                                    1, self.history_len])

            obs_state_history_normalized = (
                obs_state_history_n_normalized - obs_state_mean) / np.sqrt(obs_state_var + 1e-8)

            obs_n_norm[:, -self.history_len*(self.act_size+self.state_obs_size):-self.history_len*self.act_size] = obs_state_history_normalized

            obs_norm = obs_n_norm

            return obs_norm

    def run(self, cur_obs, last_act, obs_history, act_history):
        norm_cur_obs = self.normalize_obs(np.concatenate(
            (cur_obs, last_act)), model_type=ModelType.BASE_MODEL)
        norm_history = self.normalize_obs(np.concatenate(
            (obs_history, act_history)), model_type=ModelType.ADAP_MODULE)
        latent = self.adap_session.run(
            None, {self.adap_obs_name: norm_history})
        obs = np.concatenate((norm_cur_obs.reshape(1,-1), np.asarray(latent).reshape((1,-1))),axis=1).astype(np.float32)
        raw_act = np.asarray(self.base_session.run(None, {self.base_obs_name: obs})).squeeze()
        norm_action = (raw_act * self.act_std + self.act_mean)[0, :]
        return norm_action, raw_act


class QuadState:
    def __init__(self):

        # time
        self.t = 0

        # position
        self.pos = np.array([0, 0, 0], dtype=np.float32)

        # quaternion [w,x,y,z]
        self.att = np.array([1, 0, 0, 0], dtype=np.float32)

        # velocity
        self.vel = np.array([0, 0, 0], dtype=np.float32)

        # angular velocity i.e. body rates
        self.omega = np.array([0, 0, 0], dtype=np.float32)

        # proper acceleration i.e. acceleration - G_vec
        self.proper_acc = np.array([0.0, 0.0, 0.0])

        # commanded mass-normalized thrust, from a high-level controller
        self.cmd_collective_thrust = np.array([0.0])

        # commanded angular velocity i.e. body rates, from a high-level controller
        self.cmd_bodyrates = np.array([0.0, 0.0, 0.0])

    def __repr__(self):
        repr_str = "QuadState:\n" \
                   + " t:     [%.2f]\n" % self.t \
                   + " pos:   [%.2f, %.2f, %.2f]\n" % (self.pos[0], self.pos[1], self.pos[2]) \
                   + " att:   [%.2f, %.2f, %.2f, %.2f]\n" % (self.att[0], self.att[1], self.att[2], self.att[3]) \
                   + " vel:   [%.2f, %.2f, %.2f]\n" % (self.vel[0], self.vel[1], self.vel[2]) \
                   + " omega: [%.2f, %.2f, %.2f]\n" % (self.omega[0], self.omega[1], self.omega[2])\
                   + " proper_acc: [%.2f, %.2f, %.2f]\n" % (self.proper_acc[0], self.proper_acc[1], self.proper_acc[2])\
                   + " cmd_collective_thrust: [%.2f]\n" % (self.cmd_collective_thrust[0])\
                   + " cmd_bodyrates: [%.2f, %.2f, %.2f]\n" % (self.cmd_bodyrates[0], self.cmd_bodyrates[1], self.cmd_bodyrates[2])
        return repr_str

    