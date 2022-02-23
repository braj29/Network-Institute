import gym
from gym.utils.play import play
import griddly
from griddly import GymWrapperFactory

# This is what to use if you want to use OpenAI gym environments
#wrapper = GymWrapperFactory()
import keyboard
#rapper.build_gym_from_yaml('SokobanTutorial', 'sokoban.yaml', level=0)
keyboard.hook()
# Create the Environment
env = gym.make(f'GDY-Labyrinth-v0')
env.reset()
# # Play the game
# play(env, fps=10, zoom=1)

for _ in range(100000):

    # Qagent = pd.read_pickle(r'q_learning_oracle.pkl')
    # prob = Qagent.action_prob(state)
    # print(object)
    # action = np.random.choice([i for i in range(env.action_space.n)],p = prob/sum(prob))
    action = 2 #env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
    # print(obs)
    env.render(observer='global')  # Renders the environment from the perspective of a single player
    # time.sleep(1)
    # env.render(observer='global') # Renders the entire environment
    state = obs
    if done:
        env.reset()