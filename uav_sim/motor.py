from __future__ import print_function, division

import numpy as np
from py3dmath import Vec3  # get from https://github.com/muellerlab/py3dmath

class Motor:
    def __init__(self, position, rotAxis, minSpeed, maxSpeed, speedSqrToThrust, 
                 speedSqrToTorque, timeConst, inertia, tilt_angle=0.0):
        
        self.spinDir = np.sign(rotAxis.z)
        
        if tilt_angle == 0.0:
            self._rotAxis = rotAxis
            self._thrustAxis = Vec3(rotAxis)
            if self._thrustAxis.z < 0:
                self._thrustAxis = - self._thrustAxis
        else:
            self._rotAxis = Vec3(np.sin(tilt_angle),0,np.cos(tilt_angle)) * self.spinDir
            self._thrustAxis = Vec3(np.sin(tilt_angle),0,np.cos(tilt_angle))
            
            
        self._minSpeed = minSpeed
        self._maxSpeed = maxSpeed
        self._speedSqrToThrust = np.abs(speedSqrToThrust)
        self._speedSqrToTorque = np.abs(speedSqrToTorque)
        self._timeConst = timeConst
        self._inertia = inertia

        self._speed = None #self._minSpeed
        
        self._position = position
        
        self._thrust = Vec3(0,0,0)
        self._torque = Vec3(0,0,0)
        self._angularMomentum = Vec3(0,0,0)
        
        self._powerConsumptionInstantaneous = 0
        
        
    def run(self, dt, cmd, spdCmd=False): # TODO: spd cmd 
        oldSpeed = self._speed

        
        if spdCmd:
            speedCommand = cmd
        else:
            #we don't allow negative speeds
            if cmd < 0:
                cmd = 0
            #simulate_2CC as a first order system.
            # we correct for the body's angular velocity
            speedCommand = np.sqrt(cmd/self._speedSqrToThrust)
            
        if oldSpeed is None:
            oldSpeed = speedCommand
            self._speed = speedCommand

        else:

            if self._timeConst == 0:
                c = 0
            else:
                c = np.exp(-dt/self._timeConst)
            self._speed = c*self._speed + (1-c)*speedCommand
        
        #saturate _speed:
        if (self._speed) > self._maxSpeed:
            self._speed = self._maxSpeed
        if (self._speed) < self._minSpeed:
            self._speed = self._minSpeed
            
        #aerodynamic force & moment:
        self._angularMomentum = self._speed*self._inertia*self._rotAxis

        self._thrust = self._speedSqrToThrust*self._speed**2*self._thrustAxis
        self._torque = Vec3(0,0,0)
        #aero torque:
        self._torque += -self._speedSqrToTorque*self._speed*np.abs(self._speed)*self._rotAxis
        #torque from thrust acting at distance from com:
        self._torque += self._position.cross(self._thrust)
        #add moment due to acceleration of propeller:
        angularAcceleration = (self._speed - oldSpeed)/dt
        self._torque -= angularAcceleration*self._inertia*self._rotAxis

        self._powerConsumptionInstantaneous = self._speed*self._torque.norm2()
        return
        
        