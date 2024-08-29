from __future__ import division, print_function

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import numpy as np

class QuadcopterAttitudeControllerNested:
    def __init__(self, timeConstantAngleRollPitch, timeConstantAngleYaw, timeConstantRateRollPitch, timeConstantRateYaw):
        #A simple, nested controller.
        self._timeConstAngle_RP = timeConstantAngleRollPitch
        self._timeConstAngle_Y  = timeConstantAngleYaw
        self._timeConstRate_RP = timeConstantRateRollPitch
        self._timeConstRate_Y  = timeConstantRateYaw
        return

        

    def get_angular_acceleration(self, desNormThrust, curAtt, curAngVel):
        #Step 1: compute desired rates:
        # 1.1: construct a desired attitude, that matches the desired thrust direction
        desThrustDir = desNormThrust / desNormThrust.norm2()

        e3 = Vec3(0,0,1)
        angle = np.arccos(desThrustDir.dot(e3))
        rotAx = e3.cross(desThrustDir)
        n = rotAx.norm2()
        if n < 1e-6:
            #too small to care:
            desAtt = Rotation.identity()
        else:
            desAtt = Rotation.from_rotation_vector(rotAx*(angle/n))
            
        # 1.2 Compute desired rates:
        desRotVec = (desAtt*curAtt.inverse()).to_rotation_vector()

        desAngVel = Vec3(0,0,0)
        desAngVel.x = desRotVec.x/self._timeConstAngle_RP
        desAngVel.y = desRotVec.y/self._timeConstAngle_RP
        desAngVel.z = desRotVec.z/self._timeConstAngle_Y
        
        #Step 2: run the rates controller:
        # 2.1: Compute desired angular acceleration
        desAngAcc = desAngVel - curAngVel
        desAngAcc.x /= self._timeConstRate_RP
        desAngAcc.y /= self._timeConstRate_RP
        desAngAcc.z /= self._timeConstRate_Y
        
        return desAngAcc
    
    def get_angular_velocity(self, desNormThrust, curAtt, curAngVel):
        #Step 1: compute desired rates:
        # 1.1: construct a desired attitude, that matches the desired thrust direction
        desThrustDir = desNormThrust / desNormThrust.norm2()

        e3 = Vec3(0,0,1)
        angle = np.arccos(desThrustDir.dot(e3))
        rotAx = e3.cross(desThrustDir)
        n = rotAx.norm2()
        if n < 1e-6:
            #too small to care:
            desAtt = Rotation.identity()
        else:
            desAtt = Rotation.from_rotation_vector(rotAx*(angle/n))
            
        # 1.2 Compute desired rates:
        desRotVec = (desAtt*curAtt.inverse()).to_rotation_vector()

        desAngVel = Vec3(0,0,0)
        desAngVel.x = desRotVec.x/self._timeConstAngle_RP
        desAngVel.y = desRotVec.y/self._timeConstAngle_RP
        desAngVel.z = desRotVec.z/self._timeConstAngle_Y

        
        return desAngVel
        
        
