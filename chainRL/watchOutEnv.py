from envInterface import EnvInterface
import numpy as np
import abc

class WatchOutEnv( EnvInterface ):
    ''' Watch-Out game. In this game the agent gets a warning that something bad
        is about to happen, and has to act on it. The state is a two dimentional vector:
        ( X, Y ) where X is the inventory that the agent holds, and Y is the distance from crash
        0 <= X <= 2
        0 <= Y <= 3
        For every turn the agent gets reward equal to her inventory (X), unless Y=0, then 
        if X>0 the reward is -10, otherwise 0
        In every turn the agent can reduce or increase the inventory by 1
        In the steady state Y=3. Every turn that Y=3, in probability 5% Y will start the following 
        sequence: 2->1->0->3

        The agent has to be on X=2 all the time, and when the signal arrives - Y=2, she has to watch
        out, and reduce inventory

    '''
    def __init__( self, nTurns = 1000, crashProb = 0.01, nX = 4, nY = 6, hit = -40 ):
        self.nTurns    = nTurns
        self.crashProb = crashProb
        self.nX        = nX
        self.nY        = nY
        self.hit       = hit
        self.reset()
    
    def _getState( self ):
        return self.Y * self.nX + self.X

    def reset(self):
        self.turn = 0
        self.X    = 0
        self.Y    = self.nY - 1
        return self._getState()

    def getNumStates(self):
        return self.nX * self.nY ## dim(X) * dim(Y)

    @abc.abstractmethod
    def getNumActions(self):
        return 2

    @abc.abstractmethod
    def step(self, action ):

        ## update X
        if action: ## 1 is UP
            self.X = min( self.nX - 1, self.X + 1 )
        else:
            self.X = max( 0, self.X - 1 )

        ## update Y
        if self.Y < ( self.nY - 1 ) or ( np.random.rand() < self.crashProb ):
            self.Y = ( ( self.Y - 1 ) % self.nY )

        ## update reward
        if ( not self.Y ) and self.X > 0:
            reward = self.hit
        else:
            reward = self.X

        ## update turn
        self.turn += 1

        return ( reward, self._getState(), self.turn >= self.nTurns )




