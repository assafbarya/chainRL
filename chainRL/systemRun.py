from system import System
from qZeroAgent import QZeroAgent
from qLambdaAgent import QLambdaAgent
#from neuralAgent import NeuralAgent
from chainEnv import ChainEnv


def main():
    sys = System( ChainEnv, QLambdaAgent, 20 )
    sys.analyzeAgent()







main()