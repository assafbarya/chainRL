import numpy as np
from matplotlib import pyplot as plt

class System( object ):
    """runs the agent with the env, analyses results"""

    def __init__( self, envType, agentType, numGames ):
        self.env      = envType()
        numActions    = self.env.getNumActions()
        numStates     = self.env.getNumStates()
        self.agent    = agentType( numStates, numActions )
        self.numGames = numGames

    def playGame( self ):
        ''' plays a single game and returns the average return per turn '''
        initialState  = self.env.reset()
        self.agent.setState( initialState )
        isDone      = False
        numTurns    = 0
        totalReward = 0.
        while not isDone:
            action                    = self.agent.getAction()
            reward, nextState, isDone = self.env.step( action )
            numTurns                 += 1
            totalReward              += reward
            self.agent.update( nextState, reward )

        return totalReward / numTurns

    def analyzeAgent( self ):
        avgGameScore = np.zeros( self.numGames )
        for game in range( self.numGames ):
            avgGameScore[ game ] = self.playGame()

        self.agent.printInternalState()

        plt.plot( avgGameScore )
        plt.show()









