# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from Qagent import CartpoleAgentQ
from NNagent import CartpoleAgentNN
import pickle

def test_discretise(agent):
    fake_obs = agent.env.reset()
    discretised_obs = agent.discretise(fake_obs)
    print(f"fake obs = {fake_obs}, discretised = {discretised_obs}")

    fake_obs = [-3.4, 0.1, -0.2, 5.7]
    discretised_obs = agent.discretise(fake_obs)
    print(f"fake obs = {fake_obs}, discretised = {discretised_obs}")

    fake_obs = [4.8, 10, -0.418, 10]
    discretised_obs = agent.discretise(fake_obs)
    print(f"fake obs = {fake_obs}, discretised = {discretised_obs}")

def test_agent(agent):
    test_discretise(agent)

if __name__ == '__main__':
    #agent = CartpoleAgentQ(num_iter=20000)
    agent = CartpoleAgentNN(num_iter = 25)



    agent.learn()
    agent.show(10)

    # agent.show(10)
    #
    # agent.show(10)
    # with open('NNagent', 'wb') as file:
    #     pickle.dump(agent, file)


    # agent.plot_learning()


