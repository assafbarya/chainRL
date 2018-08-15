import abc

class EnvInterface:
    ''' defines the interface that each environment (game) has to implement'''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def getNumStates(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def getNumActions(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError()

