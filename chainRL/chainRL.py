
import numpy as np
import scipy.stats
from enum import Enum

from env import Env

class AgentModelBased:

    def description( self ):
        return 'A model based agent'

    def __init__( self, env, discountFactor = 0.95, epochs = 100 ):
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

    def __init__( self, env, discountFactor = 0.95, learningRate = 0.5, eps = 0.9, decay = 0.999, epochs = 100 ):
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


class AgentTDLambda:

    def description( self ):
        return 'TD-Lambda Agent'

    def __init__( self, env, discountFactor = 0.95, learningRate = 0.5, eps = 0.9, decay = 0.999, epochs = 100, lambda_ = 0.4 ):
        self.env            = env
        self.discountFactor = discountFactor
        self.epochs         = epochs
        self.eps            = eps
        self.decay          = decay
        self.learningRate   = learningRate
        self.lambda_        = lambda_ 

    def learnPolicy( self ):
        numStates   = self.env.getNumStates()
        Q           = np.zeros( ( numStates, 2 ) )
        E           = np.zeros( ( numStates, 2 ) )

        epoch = 0
        s = self.env.reset()
        while epoch < self.epochs:

            # decide on a step
            beGreedy           = ( np.random.rand() > self.eps ) and ( Q[ s ].sum() > 0 )
            self.eps          *= self.decay
            if beGreedy:
                a              = Q[ s ].argmax()
            else:
                a              = np.random.randint( 0, 2 )

            
            # make a step
            r, sp, isDone      = self.env.step( a )
            if isDone:
                epoch         += 1
                s              = self.env.reset()

            ## calculate q
            E[ s, a ]         += 1
            inc                = self.learningRate * ( r + self.discountFactor * Q[ sp ].max() - Q[ s, a ] )
            Q                 += inc * E
            E                 *= self.lambda_ * self.discountFactor
            s                  = sp

        return Q


def main():
    e = Env()

    for agentType in [ AgentModelBased, AgentTD0, AgentTDLambda ]:
        e.reset()
        agent = agentType( e )
        print ( agent.description() )
        print ( agent.learnPolicy() )
        print ( 'Test' )





main()