from system import System
from qZeroAgent import QZeroAgent
from qLambdaAgent import QLambdaAgent
from env import Env


def main():
    sys = System( Env, QLambdaAgent, 100 )
    sys.analyzeAgent()







main()