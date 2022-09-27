import gym

def make_env(env_name='LunarLander-v2'):
    """
    Make gym environment

    Args:
        env_name: Environment name you want to make
        seed: Random seed

    Returns:
        env: gym environment
    """
    env = gym.make(env_name)
    return env