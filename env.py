import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torchvision.transforms as trans

def make_env(env_name='BreakoutNoFrameskip-v4', render_mode=None):
    """
    Make gym environment

    Args:
        env_name: Environment name you want to make
        seed: Random seed

    Returns:
        env: gym environment
    """
    if render_mode is None:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode=render_mode)
    return env

class env_wrapper(gym.Env):
    def __init__(self, env=None):
        self.env = env
        self.preprocess = trans.Compose([trans.ToPILImage(),
                                        trans.Grayscale(), # RGB -> gray
                                        trans.Resize((110,84)), # Resize to (110,84)
                                        trans.Lambda(self.bottom_crop), # image bottom crop
                                        trans.ToTensor()]) # pixel value 0~255 -> 0~1

    def bottom_crop(self, image):
        return trans.functional.crop(image, 110-84, 0, 84, 84)

    def reset(self, seed=None):
        """
        Reset envinroment and return preprocessed state

        Args:
            seed: Random seed number

        Returns:
            unified_state: Numpy array of preprocessed first 4 frame observations which shape [4, 84, 84]
        """
        if seed:
            state = self.env.reset(seed=seed)
        else:
            state = self.env.reset()
        unified_state = self.preprocess_state(state.transpose(2, 0, 1))
        for _ in range(3):
            state, _, _, _ = self.env.step(1) # action is create ball
            preprocess_state = self.preprocess_state(state.transpose(2, 0, 1))
            unified_state = np.concatenate((unified_state, preprocess_state), axis=0)
        return unified_state

    def step(self, action):
        """
        Execute action in the trainer and return the results.

        Args:
            action: The action in action space, i.e. the output of the our agent

        Returns:
            unified_state: Numpy array of preprocessed current observation which shape [4, 84, 84]
            total_reward: Sum of rewards of 4 frames
            done: If True, the episode is over
            info: A dictionary with additional debugging information
        """
        total_reward, done = 0.0, False
        for idx in range(4):
            if done:
                preprocess_state = np.zeros((1, 84, 84))
            else:
                state, reward, done, info = self.env.step(action)
                preprocess_state = self.preprocess_state(state.transpose(2, 0, 1))
                total_reward += reward

            if idx == 0:
                unified_state = preprocess_state
            else:
                unified_state = np.concatenate((unified_state, preprocess_state), axis=0)
        return unified_state, total_reward, done, info

    def preprocess_state(self, state):
        """
        Preprocess of raw state
        - RGB -> grayscale
        - Resize (210,160) -> (110,84)
        - Bottom crop (110,84) -> (84,84)
        - Pixel value [0, 255] -> [0.0, 1.0]

        Args:
            state: Raw input state

        Returns:
            preprocessed_state: Numpy array of preprocessed state shape of [1, 84, 84]    
        """
        state_tensor = torch.from_numpy(state)
        preprocessed_state_tensor = self.preprocess(state_tensor)
        preprocessed_state = preprocessed_state_tensor.numpy()
        return preprocessed_state