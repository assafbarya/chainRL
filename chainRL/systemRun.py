from system import System
from qZeroAgent import QZeroAgent
from qLambdaAgent import QLambdaAgent
from neuralAgent import NeuralAgent
from env import Env


def main():
    sys = System( Env, NeuralAgent, 200 )
    sys.analyzeAgent()







main()