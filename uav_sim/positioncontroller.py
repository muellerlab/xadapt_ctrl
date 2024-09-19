from __future__ import print_function, division

class PositionController:
    """A simple position controller, with acceleration as output. 
    
    Makes position behave as a second order system. 
    """
    
    def __init__(self, natFreq, dampingRatio):
        self._natFreq = natFreq
        self._dampingRatio = dampingRatio
        
    
    def get_acceleration_command(self, desPos, desVel, desAcc, curPos, curVel):
        return (desPos - curPos) * self._natFreq**2 \
            + (desVel - curVel) * 2 * self._natFreq * self._dampingRatio \
            + desAcc