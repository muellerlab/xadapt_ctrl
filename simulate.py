from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from py3dmath import Vec3, Rotation  
from uav_sim.vehicle import Vehicle

from uav_sim.positioncontroller import PositionController
from uav_sim.attitudecontroller import QuadcopterAttitudeControllerNested
from uav_sim.mixer import QuadcopterMixer

from xadapt_controller.utils import QuadState,Model
from xadapt_controller.controller import AdapLowLevelControl
from pyplot3d.utils import ypr_to_R
from animate import animate_quadcopter_history
import RapidQuadrocopterTrajectories.Python.quadrocoptertrajectory as quadtraj 
import pandas as pd

np.random.seed(0)

#==============================================================================
# Define the simulation
#==============================================================================
dt = 0.002  # sdifferent
endTime = 5

#==============================================================================
# Define the vehicle
#==============================================================================
mass = 0.850  # kg
Ixx = 5e-3
Iyy = 5e-3
Izz = 10e-3
Ixy = 0
Ixz = 0
Iyz = 0
omegaSqrToDragTorque = np.matrix(np.diag([0, 0, 0.00014]))  # N.m/(rad/s)**2
armLength = 0.166  # m

##MOTORS##
motSpeedSqrToThrust = 7.6e-6  # propeller coefficient
motSpeedSqrToTorque = 1.07e-7  # propeller coefficient
motInertia   = 1e-6  #inertia of all rotating parts (motor + prop) [kg.m**2]

motTimeConst = 0.001  # time constant with which motor's speed responds [s]
motMinSpeed  = 0  #[rad/s]
motMaxSpeed  = 1000  #[rad/s]

#==============================================================================
# Define the disturbance
#==============================================================================
stdDevTorqueDisturbance = 0e-3  # [N.m]

#==============================================================================
# Define the attitude controller
#==============================================================================
#time constants for the angle components:
timeConstAngleRP = 0.08  # [s]
timeConstAngleY  = 0.40  # [s]

#gain from angular velocities
timeConstRatesRP = 0.04  # [s]
timeConstRatesY  = 0.20   # [s]

#==============================================================================
# Define the position controller
#==============================================================================
disablePositionControl = False
posCtrlNatFreq = 2  # rad/s
posCtrlDampingRatio = 0.7  # -

#==============================================================================
# Define RL lowlevel controller
#==============================================================================
low_level_controller = AdapLowLevelControl()
# Initialize our quadrotor's state
cur_state = QuadState()
# Set the maximum motor speed for this quadcopter model in RPM
# Note: because the controller can adapt down to motors-level, this highest-rpm 
# can be gained directly from motor datasheet without measurements from the experiments.
low_level_controller.set_max_motor_spd(motMaxSpeed)

#==============================================================================
# Compute all things:
#==============================================================================

inertiaMatrix = np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
quadrocopter = Vehicle(mass, inertiaMatrix, omegaSqrToDragTorque, stdDevTorqueDisturbance)


# Our quadcopter model as    
#           
#           x
#           ^
#   (-)mot3 | mot0(+)
#           |
#     y<----+-----
#           |
#   (+)mot2 | mot1(-)


motor_pos = armLength*(2**0.5)

quadrocopter.add_motor(Vec3( motor_pos, -motor_pos, 0), Vec3(0,0,1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)
quadrocopter.add_motor(Vec3( -motor_pos, -motor_pos, 0), Vec3(0,0,-1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)
quadrocopter.add_motor(Vec3(-motor_pos, +motor_pos, 0), Vec3(0,0,1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)
quadrocopter.add_motor(Vec3( motor_pos,motor_pos, 0), Vec3(0,0, -1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)

posControl = PositionController(posCtrlNatFreq, posCtrlDampingRatio)
attController = QuadcopterAttitudeControllerNested(timeConstAngleRP, timeConstAngleY, timeConstRatesRP, timeConstRatesY)
mixer = QuadcopterMixer(mass, inertiaMatrix, motor_pos, motSpeedSqrToTorque/motSpeedSqrToThrust)

#==============================================================================
# Define the trajectory
#==============================================================================

# Define the trajectory starting state:
pos0 = np.random.uniform(-2, 2, 3)  # position
pos0[2] += 2 # make sure it starts above the ground
vel0 = np.random.uniform(-2,2,3) #velocity
att0 = np.random.uniform(-1,1,4) #attitude
att0 /= np.linalg.norm(att0) #normalize

# Define the goal state:
posf = [0, 0, 2]  # position
velf = [0, 0, 0]  # velocity
accf = [0, 0, 0]  # acceleration

quadrocopter.set_position(Vec3(pos0))
quadrocopter.set_velocity(Vec3(vel0))
quadrocopter.set_attitude(Rotation(att0[0], att0[1], att0[2], att0[3]))
    
#start at equilibrium rates:
quadrocopter._omega = Vec3(0,0,0)


#==============================================================================
# Run the simulation
#==============================================================================

numSteps = int((endTime)/dt)
index = 0

t = 0

posHistory       = np.zeros([numSteps,3])
velHistory       = np.zeros([numSteps,3])
angVelHistory    = np.zeros([numSteps,3])
attHistory       = np.zeros([numSteps,3])
motForcesHistory = np.zeros([numSteps,quadrocopter.get_num_motors()])
inputHistory     = np.zeros([numSteps,quadrocopter.get_num_motors()])
times            = np.zeros([numSteps,1])

while index < numSteps:
    #define commands:
    accDes = posControl.get_acceleration_command(Vec3(posf), Vec3(velf), Vec3(accf),
                                                 quadrocopter._pos, quadrocopter._vel)
    if disablePositionControl:
        accDes *= 0 #disable position control
        
        
    ########################################## Model-based low-level Control  ##########################################################
    # #mass-normalised thrust:
    # thrustNormDes = accDes + Vec3(0, 0, 9.81)
    # angAccDes = attController.get_angular_acceleration(thrustNormDes, quadrocopter._att, quadrocopter._omega)
    # motCmds = mixer.get_motor_force_cmd(thrustNormDes, angAccDes)
    
    # #run the simulator
    # quadrocopter.run(dt, motCmds)
    ########################################## RL Control ##########################################################
    #mass-normalised thrust:
    thrustNormDes = accDes + Vec3(0, 0, 9.81)
    #desired ang velocity 
    angVelDes = attController.get_angular_velocity(thrustNormDes, quadrocopter._att, quadrocopter._omega)
    cur_state.att = quadrocopter._att.to_array().flatten()
    cur_state.omega = quadrocopter._omega.to_array().flatten()
    cur_state.proper_acc = quadrocopter._accel.to_array().flatten()
    cur_state.cmd_collective_thrust = thrustNormDes.z 
    cur_state.cmd_bodyrates = angVelDes.to_array().flatten()
    
    motCmds = low_level_controller.run(cur_state)
    #run the simulator 
    quadrocopter.run(dt, motCmds,spdCmd=True)

    #for plotting
    times[index] = t
    inputHistory[index,:]     = motCmds #motForceCmds
    posHistory[index,:]       = quadrocopter._pos.to_list()
    velHistory[index,:]       = quadrocopter._vel.to_list()
    attHistory[index,:]       = quadrocopter._att.to_euler_YPR()
    angVelHistory[index,:]    = quadrocopter._omega.to_list()
    motForcesHistory[index,:] = quadrocopter.get_motor_forces()

    t += dt
    index += 1

#==============================================================================
# Make the plots
#==============================================================================
   
fig, ax = plt.subplots(5,1, sharex=True)

ax[0].plot(times, posHistory[:,0], label='x')
ax[0].plot(times, posHistory[:,1], label='y')
ax[0].plot(times, posHistory[:,2], label='z')
ax[0].plot(times, np.ones_like(times)*posf[0], '--', color='C0')
ax[0].plot(times, np.ones_like(times)*posf[1], '--', color='C1')
ax[0].plot(times, np.ones_like(times)*posf[2], '--', color='C2')
ax[1].plot(times, velHistory)
ax[2].plot(times, attHistory[:,0]*180/np.pi, label='Y')
ax[2].plot(times, attHistory[:,1]*180/np.pi, label='P')
ax[2].plot(times, attHistory[:,2]*180/np.pi, label='R')
ax[3].plot(times, angVelHistory[:,0], label='p')
ax[3].plot(times, angVelHistory[:,1], label='q')
ax[3].plot(times, angVelHistory[:,2], label='r')
ax[4].plot(times, inputHistory)
ax[4].plot(times, inputHistory)

ax[-1].set_xlabel('Time [s]')

ax[0].set_ylabel('Pos')
ax[1].set_ylabel('Vel')
ax[2].set_ylabel('Att [deg]')
ax[3].set_ylabel('AngVel (in B)')
ax[4].set_ylabel('MotForces')

ax[0].set_xlim([0, endTime])
ax[0].legend()
ax[2].legend()
ax[3].legend()

plt.show()

data_dict = {
    'Time': times.flatten(),
    'PosX': posHistory[:, 0],
    'PosY': posHistory[:, 1],
    'PosZ': posHistory[:, 2],
    'VelX': velHistory[:, 0],
    'VelY': velHistory[:, 1],
    'VelZ': velHistory[:, 2],
    'Yaw': attHistory[:, 0],
    'Pitch': attHistory[:, 1],
    'Roll': attHistory[:, 2],
}
# Create a DataFrame from the dictionary
data = pd.DataFrame(data_dict)

# Save the DataFrame to a CSV file
data.to_csv('data/quadcopter_data.csv', index=False)


# Load your data 
data = pd.read_csv('data/quadcopter_data.csv')
times = data['Time'].values

posHistory = data[['PosX', 'PosY', 'PosZ']].values
attHistory = data[['Yaw','Pitch', 'Roll']].values
x = posHistory.T
steps = len(times)

R = np.zeros((3, 3, steps))
for i in range(steps):
    ypr = attHistory[i,:]
    R[:, :, i] = ypr_to_R(ypr, degrees=False)


animate_quadcopter_history(times, x, R, arm_length=0.4)