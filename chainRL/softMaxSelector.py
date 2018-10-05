from actionSelectorInterface import ActionSelectorInterface
import numpy as np
import warnings


class SoftMaxSelector( ActionSelectorInterface ):
    """softmax type action selector"""
    def __init__( self, temp = 0.5, coolingCoeff = 0.9999, minTemp = 0.01 ):
        self.temp         = temp
        self.coolingCoeff = coolingCoeff
        self.minTemp      = minTemp

    def getAction( self, actionValues ):
        ## act randomly until all actions have non zero values, or if all values are equal
        ## since in the softmax function it will have zero prob
        if ( actionValues.prod() == 0 ) or ( actionValues.var() == 0 ):
            return np.random.randint( actionValues.size )
        warnings.simplefilter("error", RuntimeWarning)
        try:
            normVals   =  actionValues / actionValues.max()
            weights    = np.exp( normVals / self.temp )
            probs      = weights / weights.sum()
            cs         = np.concatenate( [ np.array( [ 0 ] ), probs.cumsum()[ : -1 ] ] )
            r          = np.random.rand()
            cs[cs>r]   = -1
            self.temp = max( self.temp * self.coolingCoeff, self.minTemp )
            return cs.argmax()
        except RuntimeWarning as w:
            print ('warning', w)
            return 0


        





