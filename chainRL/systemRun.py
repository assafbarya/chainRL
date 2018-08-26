from system import System
from qZeroAgent import QZeroAgent
from qLambdaAgent import QLambdaAgent
#from neuralAgent import NeuralAgent
from chainEnv import ChainEnv
from watchOutEnv import WatchOutEnv
from exploreThenExploitAgent import ExploreThenExploitAgent

def main():
    sys = System( WatchOutEnv, ExploreThenExploitAgent, 100 )
    sys.analyzeAgent()







main()