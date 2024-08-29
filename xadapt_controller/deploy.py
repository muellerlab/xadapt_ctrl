from utils import QuadState,Model
from controller import AdapLowLevelControl

# Initialize our adaptive low level controller
low_level_controller = AdapLowLevelControl()

# Initialize our quadrotor's state
cur_state = QuadState()

# Set the maximum motor speed for this quadcopter model in RPM
low_level_controller.set_max_motor_spd(3000)

# Run the controller to output motorspeed commands
# Our quadcopter model as    
#           
#           x
#           ^
#      mot3 | mot0
#           |
#     y<----+-----
#           |
#      mot2 | mot1
#    
motor_spd_command = low_level_controller.run(cur_state)
