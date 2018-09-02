from agentInterface import AgentInterface
import numpy as np

class QLambdaAgent( AgentInterface ):
    """Q(lambda) Agent"""
    def description( self ):
        return 'Q(lambda) Agent'

    def __init__(   self, 
                    numStates, 
                    numActions, 
                    actionSelectorType, 
                    discountFactor = 0.95, 
                    learningRate   = 0.1, 
                    lambda_        = 0.3 ):

        self.discountFactor = discountFactor
        self.numActions     = numActions
        self.learningRate   = learningRate
        self.Q              = np.zeros( ( numStates, numActions ) )
        self.E              = np.zeros( ( numStates, numActions ) )
        self.lastAction     = None
        self.lambda_        = lambda_ 
        self.actionSelector = actionSelectorType()

    def setState( self, state ):
        self.state          = state

    def getAction( self ):
        action                        = self.actionSelector.getAction( self.Q[ self.state ] )
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


