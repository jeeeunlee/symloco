import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

BOOL_LOG = False
class TimePrinter:
    def __init__(self, mode: str):
        self.t0 = time.time()
        self.mode = mode

    def print(self, log: str):
        global BOOL_LOG
        t = time.time()
        if BOOL_LOG:
            print(f"{log} took {t-self.t0}s ")
        if self.mode == 'reset':
            self.t0 = time.time()


log = TimePrinter(mode='reset')

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)
log.print("make_vec_env")

model = PPO("MlpPolicy", vec_env, verbose=1)
log.print("model")

model.learn(total_timesteps=100000) #25000
log.print("learn")

model.save("ppo_cartpole")
log.print("save")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")
log.print("load")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    t2 = time.time()
    log.print("save")