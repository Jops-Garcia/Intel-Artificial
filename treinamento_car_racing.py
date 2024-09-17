import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

env = gym.make("CarRacing-v2", render_mode="rgb_array")

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="log",
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

eval_env = gym.make("CarRacing-v2", render_mode="rgb_array")

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=5,
    deterministic=False,
)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

model.learn(total_timesteps=int(4e5), log_interval=10, progress_bar=False)

model.save("PPO5_CarRacing_"+str(int(400000)))

model = PPO.load("PPO1_CarRacing_400000", env=eval_env)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=False)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

import base64
from pathlib import Path

from IPython import display as ipythondisplay


def show_videos(video_path="", prefix=""):
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

def record_video(env_id, model, video_length=1000, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make("CarRacing-v2", render_mode="rgb_array")])

    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)


    eval_env.close()

record_video("CarRacing-v2", model, video_length=1000, prefix="ppo2-carracing")

show_videos("videos", prefix="ppo2")