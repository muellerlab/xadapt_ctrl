#!/usr/bin/python3
import numpy as np
from .utils import Model
from scipy.spatial.transform import Rotation as R
from collections import deque
    
# inputs current raw data (human-readable, non-normalized), outputs actual motor speed
class AdapLowLevelControl:
    def __init__(self):

        # time
        self.t = 0
        # Learning-based controller
        self.model = Model()

        
        self.maxMotorSpd = 5000 
        
        self.state_vars = ['ori1', 'ori2', 'ori3', 'ori4', 'ori5', 'ori6', 'ori7', 'ori8', 'ori9', 'wx', 'wy', 'wz', 'prop_acc', 'cmd_wx', 'cmd_wy', 'cmd_wz', 'cmd_prop_acc']
        self.action_vars = ['act1', 'act2', 'act3', 'act4']
        
        history_len = 400
        act_size = len(self.action_vars)
        state_obs_size = len(self.state_vars)
        
        self.cur_obs = np.zeros((state_obs_size,))
        self.last_act = np.zeros((act_size,))
        
        
        self.model.set_const_sizes(state_obs_size,act_size,history_len)
        
        self.obs_history = deque([np.zeros(state_obs_size)]*history_len)

        self.act_history = deque([np.zeros(act_size)]*history_len)
        
        self.model.activate()
        
    def set_max_motor_spd(self,max_motor_spd):
        self.maxMotorSpd = max_motor_spd
        
    def convert_vehState(self,veh_state):
        att_aray = np.array([veh_state.att[1], veh_state.att[2],
                             veh_state.att[3], veh_state.att[0]])
        rotation_matrix = R.from_quat(
            att_aray).as_matrix().reshape((9,), order="F")
        cur_obs = np.concatenate((rotation_matrix, 
                                          veh_state.omega, 
                                          np.array([veh_state.proper_acc[2]],dtype=np.float32),  
                                          veh_state.cmd_bodyrates,
                                          np.array([veh_state.cmd_collective_thrust],dtype=np.float32),  
                                                    ), axis=0).astype(np.float32)
        return cur_obs
        
    def run(self,veh_state):
        cur_obs = self.convert_vehState(veh_state)
        
        norm_act, raw_act = self.model.run(cur_obs,
            self.last_act,np.asarray(self.obs_history, dtype=np.float32).flatten(),
            np.asarray(self.act_history, dtype=np.float32).flatten())
        
        
        self.obs_history.popleft()
        self.obs_history.append(cur_obs)
        self.act_history.popleft()
        self.act_history.append(raw_act)
        
        self.last_act = raw_act
        
        # Drone model is    
        #           
        #           x
        #           ^
        #      mot3 | mot0
        #           |
        #     y<----+-----
        #           |
        #      mot1 | mot2
        #    
        
        spd_cmd = norm_act.squeeze() * self.maxMotorSpd
        
        # hardcode to fit simulate drone model
        temp = spd_cmd[2]
        spd_cmd[2] = spd_cmd[1]
        spd_cmd[1] = temp
        
        return spd_cmd