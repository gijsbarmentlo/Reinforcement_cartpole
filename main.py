# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from Qagent import CartpoleAgentQ
from NNagent import CartpoleAgentNN
import pickle
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

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

    #tf.config.list_physical_devices
    agent = CartpoleAgentNN(num_iter = 100)


    for i in range(10):
        print(i)
        agent.run()
        agent.show(5)
        agent.memory_replay()
        agent.show(5)

    #agent.plot_learning()

#    agent.memory_replay()


    # with open('NNagent', 'wb') as file:
    #     pickle.dump(agent, file)




