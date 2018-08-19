from agentInterface import AgentInterface
import numpy as np

class QZeroAgent( AgentInterface ):
    """Q(0) Agent"""
    def description( self ):
        return 'Q(0) Agent'

    def __init__( self, numStates, numActions, discountFactor = 0.97, learningRate = 0.3, eps = 0.6, decay = 0.999 ):
        self.discountFactor = discountFactor
        self.numActions     = numActions
        self.eps            = eps  ## epsilon greedy coefficient
        self.decay          = decay ## decay of the epsilon greedy
        self.learningRate   = learningRate
        self.Q              = np.zeros( ( numStates, numActions ) )
        self.lastAction     = None

    def setState( self, state ):
        self.state          = state

    def getAction( self ):
        beGreedy            = ( np.random.rand() > self.eps ) and ( self.Q[ self.state ].sum() > 0 )
        self.eps           *= self.decay
        if beGreedy: 
            action          = self.Q[ self.state ].argmax()
        else: 
            action          = np.random.randint( self.numActions )

        self.lastAction     = action
        return action

    def update( self, nextState, reward ):
        self.Q[ self.state, self.lastAction ] = ( 1 - self.learningRate ) * self.Q[ self.state, self.lastAction ] \
                            + self.learningRate * ( reward + self.discountFactor * self.Q[ nextState ].max() )

        self.state = nextState

    def printInternalState( self ):
        print( self.Q )


