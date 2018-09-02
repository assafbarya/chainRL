import abc

class ActionSelectorInterface(object):
    """action selector interface"""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getAction( self, actionValues ):
        ''' given a vector of actionValues, select an action '''
        raise NotImplementedError()



