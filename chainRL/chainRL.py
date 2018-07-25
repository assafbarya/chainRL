
import numpy as np
import scipy.stats
from enum import Enum

#class Action( Enum ):
#    DOWN = 0
#    UP   = 1

#    #STAY = 2

class Env:
    ''' Chain game, with nStates states. 0,1,2,..nStates-1
        At each point the player can chose to go up or down
        With errProb, the player goes in the opposite direction of the intentional one
        Going down brings the player back to state 0, collecting downReward
        Going up brings the player one state up, with no reward
        When reaching the final state, the reward for going up is finalReward
    '''

    def __init__( self, nStates = 5,
                  nTurns        = 1000,
                  downReward    = 2,
                  finalReward   = 10,
                  errProb       = 0.01 ):
        ''' init '''
        self.nStates     = nStates
        self.nTurns      = nTurns
        self.downReward  = downReward
        self.finalReward = finalReward
        self.errProb     = errProb
        self.state       = 0
        self.maxState    = nStates - 1
        self.turn        = 0

    def reset( self ):
        self.state  = 0
        self.turns = 0
        return self.state

    def getNumStates( self ):
        return self.nStates

    def step( self, a ):
        isErr = np.random.rand() < self.errProb
        if ( a == 1 ) ^ isErr: ## selected action is UP
            r          = self.finalReward if self.state == self.maxState else 0
            self.state = min( self.maxState, self.state + 1 )
        else:
            self.state = 0
            r          = 2
        self.turn += 1

        return ( r, self.state, self.turn >= self.nTurns )

class Agent:
    def __init__( self, env, discountFactor = 0.9, learningRate = 0.1, epochs = 1000 ):
        self.env            = env
        self.discountFactor = discountFactor
        self.learningRate   = learningRate
        self.epochs         = epochs
        self.qMat           = np.zeros( ( env.getNumStates(), 2 ) )

    def learnPolicy( self ):
        for _ in range( self.epochs ):
            isDone = False
            s = self.env.reset()
            while not isDone:
                a = np.random.randint( 0, 2 )
                r, newState, isDone = self.env.step( a )
                self.qMat[ s, a ] = ( 1 - self.learningRate ) * self.qMat[ s, a ] \
                                    + self.learningRate * ( r + self.discountFactor * self.qMat[ newState ].max() )
                s = newState

            return self.qMat


class Agent2:
    def __init__( self, env, discountFactor = 0.9, learningRate = 0.1, epochs = 1000 ):
        self.env            = env
        self.discountFactor = discountFactor
        self.epochs         = epochs

    def learnPolicy( self ):
        numStates   = self.env.getNumStates()
        states      = list( range( numStates ) )
        rSum        = np.zeros( ( numStates, 2, numStates ) ) ## sum of rewards collected
        transitions = np.zeros( ( numStates, 2, numStates ) ) ## counts transitions

        ## collect rewards information
        for _ in range( self.epochs ):
            isDone = False
            s = self.env.reset()
            while not isDone:
                a                        = np.random.randint( 0, 2 )
                r, sp, isDone            = self.env.step( a )
                transitions[ s, a, sp ] += 1
                rSum[ s, a, sp ]        += r
                s                        = sp
        r = rSum / transitions 
        r[np.isnan(r)]=0

        ## calculate transition probability matrtix
        p = transitions 
        for s in states:
            for a in [ 0, 1 ]:
                sum = transitions[ s, a ].sum()
                for sp in states:
                    p[ s, a, sp ] /= sum

        p[np.isnan(p)]=0

        ## calculate q
        Q = np.zeros( ( numStates, 2 ) )
        for _ in range( self.epochs  ):
            newQ = np.zeros( ( numStates, 2 ) )
            for s in states:
                for a in [ 0, 1 ]:
                    for sp in states:
                        newQ[ s, a ] += p[ s, a, sp ] * ( r[ s, a, sp ] + self.discountFactor * Q[ sp ].max() )
            Q = newQ

        return Q




def main():
    e = Env()
    agent = Agent2( e )
    print ( agent.learnPolicy() )






main()