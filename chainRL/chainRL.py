
import numpy as np
import scipy.stats
from enum import Enum


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
                  errProb       = 0.1 ):
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
        self.turn   = 0
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


class AgentExploreThenCalculate:

    def description( self ):
        return 'An agent that first explores all the state-action-reward space, and then finds the optimal policy'

    def __init__( self, env, discountFactor = 0.9, epochs = 100 ):
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
        for _ in range( self.epochs ):
            newQ = np.zeros( ( numStates, 2 ) )
            for s in states:
                for a in [ 0, 1 ]:
                    for sp in states:
                        newQ[ s, a ] += p[ s, a, sp ] * ( r[ s, a, sp ] + self.discountFactor * Q[ sp ].max() )
            Q = newQ

        return Q


class AgentTD0:

    def description( self ):
        return 'TD-0 Agent'

    def __init__( self, env, discountFactor = 0.9, learningRate = 0.5, eps = 0.9, decay = 0.999, epochs = 100 ):
        self.env            = env
        self.discountFactor = discountFactor
        self.epochs         = epochs
        self.eps            = eps
        self.decay          = decay
        self.learningRate   = learningRate

    def learnPolicy( self ):
        numStates   = self.env.getNumStates()
        Q           = np.zeros( ( numStates, 2 ) )

        epoch = 0
        s = self.env.reset()
        while epoch < self.epochs:
            beGreedy           = ( np.random.rand() > self.eps ) and ( Q[ s ].sum() > 0 )
            self.eps          *= self.decay
            if beGreedy:
                a              = Q[ s ].argmax()
            else:
                a              = np.random.randint( 0, 2 )

            r, sp, isDone      = self.env.step( a )
            if isDone:
                epoch         += 1
                s              = self.env.reset()

            ## calculate q
            Q[ s, a ] = ( 1 - self.learningRate ) * Q[ s, a ] \
                              + self.learningRate * ( r + self.discountFactor * Q[ sp ].max() )

            s                  = sp

        return Q




def main():
    e = Env()

    for agentType in [ AgentExploreThenCalculate, AgentTD0 ]:
        e.reset()
        agent = agentType( e )
        print ( agent.description() )
        print ( agent.learnPolicy() )






main()