from system import System
from qZeroAgent import QZeroAgent
from qLambdaAgent import QLambdaAgent
from neuralAgent import NeuralAgent
from chainEnv import ChainEnv
from watchOutEnv import WatchOutEnv


def main():
    sys = System( WatchOutEnv, QLambdaAgent, 20 )
    sys.analyzeAgent()







main()