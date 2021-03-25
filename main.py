# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from Qagent import CartpoleAgentQ
from NNagent import CartpoleAgentNN

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    agent = CartpoleAgentNN()

    for i in range(15):
        print(i)
        agent.run(num_episodes=50)
        agent.show(10)

    agent.plot_learning()


