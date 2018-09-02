from agentInterface import AgentInterface
from keras import Sequential
from keras.layers import InputLayer, Dense
import numpy as np

class NeuralAgent( AgentInterface ):
    ''' an agent containing a neural network '''
    def description( self ):
        return 'Neural Network Agent'

    def __init__(   self, 
                    numStates, 
                    numActions, 
                    actionSelectorType, 
                    discountFactor = 0.95 ):

        self.discountFactor = discountFactor
        self.numActions     = numActions
        self.numStates      = numStates
        self.model          = Sequential()

        self.model.add( InputLayer( batch_input_shape = ( 1, numStates ) ) )
        self.model.add( Dense( 10, activation = 'sigmoid' ) )
        self.model.add( Dense( numActions, activation = 'linear' ) )
        self.model.compile( loss = 'mse', optimizer = 'adam', metrics = [ 'mae' ] )

        self.stateRep       = np.identity( numStates )
        self.actionSelector = actionSelectorType()

    def _getStateRepr( self, state ):
        return self.stateRep[ state : state + 1 ]

    def _getModelPredictions( self, state ):
        return self.model.predict( self._getStateRepr( state ) )

    def setState( self, state ):
        self.state = state

    def getAction( self ):
        action          = self.actionSelector.getAction( self._getModelPredictions( self.state ) )
        self.lastAction = action
        return action

    def update( self, nextState, reward ):
        target                        = reward + self.discountFactor * np.max( self._getModelPredictions( nextState ) )
        target_vec                    = self._getModelPredictions( self.state )[ 0 ]
        target_vec[ self.lastAction ] = target
        target_vec                    = target_vec.reshape( -1, 2 )

        self.model.fit( self._getStateRepr( self.state ), target_vec, epochs = 1, verbose = 0 )
        self.state = nextState

    def printInternalState( self ):
        for state in range( self.numStates ):
            print( self._getModelPredictions( state ) )
