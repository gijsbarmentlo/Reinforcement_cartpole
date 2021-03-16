
import gym
env = gym.make('CartPole-v0')
env.reset()
print(f"obs_space : {env.observation_space}")
print(f"action_space : {env.action_space}")

for t in range(200):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(f"t : {t} \n obs:{obs} \n reward:{reward} \n done:{done} \n \n")
env.close()

print