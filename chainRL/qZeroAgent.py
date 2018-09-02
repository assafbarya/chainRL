from agentInterface import AgentInterface
import numpy as np

class QZeroAgent( AgentInterface ):
    """Q(0) Agent"""
    def description( self ):
        return 'Q(0) Agent'

    def __init__(   self, 
                    numStates, 
                    numActions,
                    actionSelectorType, 
                    discountFactor = 0.95, 
                    learningRate = 0.1 ):

        self.discountFactor = discountFactor
        self.numActions     = numActions
        self.learningRate   = learningRate
        self.Q              = np.zeros( ( numStates, numActions ) )
        self.lastAction     = None
        self.actionSelector = actionSelectorType()

    def setState( self, state ):
        self.state          = state

    def getAction( self ):
        action          = self.actionSelector.getAction( self.Q[ self.state ] )
        self.lastAction = action
        return action

    def update( self, nextState, reward ):
        self.Q[ self.state, self.lastAction ] = ( 1 - self.learningRate ) * self.Q[ self.state, self.lastAction ] \
                            + self.learningRate * ( reward + self.discountFactor * self.Q[ nextState ].max() )

        self.state = nextState

    def printInternalState( self ):
        print( self.Q )


