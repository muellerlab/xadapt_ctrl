from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from uav_sim.motor import Motor

class Vehicle:
    def __init__(self, mass, inertiaMatrix, omegaSqrToDragTorque, disturbanceTorqueStdDev):
        self._inertia = inertiaMatrix
        self._mass = mass
        
        self._pos = Vec3(0,0,0)
        self._vel = Vec3(0,0,0)
        self._att = Rotation.identity()
        self._omega = Vec3(0,0,0)
        self._accel = Vec3(0,0,0)

        self._motors = []
        
        self._omegaSqrToDragTorque = omegaSqrToDragTorque
        
        self._disturbanceTorqueStdDev = disturbanceTorqueStdDev
        return
        

    def add_motor(self, motorPosition, spinDir, minSpeed, maxSpeed, speedSqrToThrust, speedSqrToTorque, timeConst, inertia, tilt_angle=0.0):
        self._motors.append(Motor(motorPosition, spinDir, minSpeed, maxSpeed, speedSqrToThrust, speedSqrToTorque, timeConst, inertia, tilt_angle=tilt_angle))
        return
        
        
    def run(self, dt, motorCmds,spdCmd=False): # TODO: spd cmd 
        
        
        totalForce_b  = Vec3(0,0,0)
        totalTorque_b = Vec3(0,0,0)
        for (mot,thrustCmd) in zip(self._motors, motorCmds):
            mot.run(dt, thrustCmd,spdCmd)
            
            totalForce_b  += mot._thrust
            totalTorque_b += mot._torque
        
        totalTorque_b += (- self._omega.norm2()*self._omegaSqrToDragTorque*self._omega)
        
        #add noise:
        totalTorque_b += Vec3(np.random.normal(), np.random.normal(), np.random.normal())*self._disturbanceTorqueStdDev
        
        
        angMomentum = self._inertia*self._omega
        for mot in self._motors:
            angMomentum += mot._angularMomentum

        angAcc = np.linalg.inv(self._inertia)*(totalTorque_b - self._omega.cross(angMomentum))
        
        #translational acceleration:
        acc = Vec3(0,0,-9.81)  # gravity
        acc += self._att*totalForce_b/self._mass
        
        vel = self._vel
        att = self._att
        omega = self._omega
        
        
        #euler integration
        self._pos += vel*dt
        self._vel += acc*dt
        self._att  = att*Rotation.from_rotation_vector(omega*dt)
        self._omega += angAcc*dt
        
        accMeas = (self._att.inverse() * (acc + Vec3(0, 0, 9.81))); 
        self._accel = accMeas
        
    def set_position(self, pos):
        self._pos = pos
        
        
    def set_velocity(self, velocity):
        self._vel = velocity
        
        
    def set_attitude(self, att):
        self._att = att
        
        
    def get_num_motors(self):
        return len(self._motors)
    
    
    def get_motor_speeds(self):
        out = np.zeros(len(self._motors,))
        for i in range(len(self._motors)):
            out[i] = self._motors[i]._speed
        return out


    def get_motor_forces(self):
        out = np.zeros(len(self._motors,))
        for i in range(len(self._motors)):
            out[i] = self._motors[i]._thrust.z
        return out
    
    def get_total_power_consumption(self):
        pwr = 0
        for m in self._motors:
            pwr += m._powerConsumptionInstantaneous 
            
            
        return pwr
