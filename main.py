# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from Qagent import CartpoleAgentQ

# Config
BUCKETS = [5, 7, 20, 10]
LEARNING_RATE = 0.01
DISCOUNT_RATE = 0.90
MIN_EPSILON = 0.1
DECAY = 0.8
NUM_ITER = 300
MAX_TIME = 220


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
    agent = CartpoleAgentQ(BUCKETS, LEARNING_RATE, DISCOUNT_RATE, MIN_EPSILON, DECAY, num_iter=2000, max_time=MAX_TIME)
    agent.learn()
    agent.show(10)
    agent.plot_learning()

