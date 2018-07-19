
import numpy as np
import scipy.stats
from enum import Enum

class Action( Enum ):
    UP   = 0
    DOWN = 1

class Env:
    ''' Chain game, with nStates states. 0,1,2,..nStates-1
        At each point the player can chose to go up or down
        With errProb, the player goes in the opposite direction of the intentional one
        Going down brings the player back to state 0, collecting downReward
        Going up brings the player one state up, with no reward
        When reaching the final state, the reward for going up is finalReward
    '''

    def __init__( self, nStates = 10,
                  nTurns        = 1000,
                  downReward    = 2,
                  finalReward   = 10,
                  errProb       = 0.4 ):
        ''' init '''
        self.nStates     = nStates
        self.nTurns      = nTurns
        self.downReward  = downReward
        self.finalReward = finalReward
        self.errProb     = errProb
        self.state       = 0
        self.maxState    = nStates - 1
        self.turn        = 0

    def step( self, a ):
        isErr = np.random.rand() < self.errProb
        if ( a == Action.UP ) ^ isErr: ## selected action is UP
            r          = self.finalReward if self.state == self.maxState else 0
            self.state = min( self.maxState, self.state + 1 )
        else:
            self.state = 0
            r          = 2
        self.turn += 1

        return ( r, self.state, self.turn )

        
def main():
    e = Env()
    for _x in range(100):
        print ( e.step( Action.UP ) )






main()