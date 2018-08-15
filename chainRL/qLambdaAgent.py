from agentInterface import AgentInterface
import numpy as np

class QLambdaAgent( AgentInterface ):
    """Q(lambda) Agent"""
    def description( self ):
        return 'Q(lambda) Agent'

    def __init__( self, numStates, numActions, discountFactor = 0.95, learningRate = 0.2, eps = 0.6, decay = 0.999, lambda_ = 0.2 ):
        self.discountFactor = discountFactor
        self.numActions     = numActions
        self.eps            = eps  ## epsilon greedy coefficient
        self.decay          = decay ## decay of the epsilon greedy
        self.learningRate   = learningRate
        self.Q              = np.zeros( ( numStates, numActions ) )
        self.E              = np.zeros( ( numStates, numActions ) )
        self.lastAction     = None
        self.lambda_        = lambda_ 

    def setState( self, state ):
        self.state          = state

    def getAction( self ):
        beGreedy                      = ( np.random.rand() > self.eps ) and ( self.Q[ self.state ].sum() > 0 )
        self.eps                     *= self.decay
        if beGreedy: 
            action                    = self.Q[ self.state ].argmax()
        else: 
            action                    = np.random.randint( self.numActions )

        self.lastAction               = action
        self.E[ self.state, action ] += 1

        return action

    def update( self, nextState, reward ):
        inc     = self.learningRate * ( reward + self.discountFactor * self.Q[ nextState ].max() \
                                        - self.Q[ self.state, self.lastAction ] )
        self.Q += inc * self.E
        self.E *= self.lambda_ * self.discountFactor

        self.state = nextState

    def printInternalState( self ):
        print( self.Q )


