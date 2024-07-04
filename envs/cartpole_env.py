import gym
import cv2


class WrappedEnv(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self.full_image_size = config["obs_size"]

    def reset(self, return_render=False):
        obs = self.env.reset()
        if return_render:
            obs = obs, self.env.render()
        return obs

    def step(self, action, return_render=False):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # done = terminated or truncated
        if return_render:
            next_state = next_state, self.env.render()
        return next_state, reward, terminated, truncated, info


def make_env(config, render_mode="rgb_array"):
    env = WrappedEnv(gym.make(config["env_name"], render_mode=render_mode), config)
    return env
