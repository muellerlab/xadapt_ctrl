from __future__ import division, print_function

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import numpy as np

class QuadcopterMixer:
    def __init__(self, mass, inertiaMatrix, armLength, thrustToTorque):
        #A simple, nested controller.
        self._mass          = mass
        self._inertiaMatrix = inertiaMatrix
        
        #compute mixer matrix:
        l = armLength
        k = thrustToTorque
        M = np.array([[ 1,  1,  1,  1],
                      [ -l,  -l,  l, l],
                      [-l,  l,  l,  -l],
                      [ -k, k,  -k, k],
                      ])
        
        self._mixerMat = np.linalg.inv(M)
        return

        
    def get_motor_force_cmd(self, desNormThrust, desAngAcc):
        ftot = self._mass*desNormThrust.norm2()
        moments = self._inertiaMatrix*desAngAcc
        return self._mixerMat.dot(np.array([ftot, moments.x, moments.y, moments.z]))
        
        
