from system import System
from qZeroAgent import QZeroAgent
from qLambdaAgent import QLambdaAgent
#from neuralAgent import NeuralAgent
from chainEnv import ChainEnv
from watchOutEnv import WatchOutEnv
from exploreThenExploitAgent import ExploreThenExploitAgent
from epsGreedySelector import EpsGreedySelector
from softMaxSelector import SoftMaxSelector

def main():

    sys = System( envType            = WatchOutEnv, 
                  agentType          = QLambdaAgent, 
                  numGames           = 100,
                  actionSelectorType = EpsGreedySelector )

    sys.analyzeAgent()







main()