import os
import warnings

import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from model import SAC
from env import make_env

warnings.filterwarnings("ignore", category=DeprecationWarning) # for not appearing warning messages
def test(cfg):
    """
    Test model
    Print average scores, Save playing video
    """
    ########################
    #    make SAC model    #
    ########################
    device = cfg['device']
    state_dim = cfg['env']['state_dim']
    action_dim = cfg['env']['action_dim']
    hidden_dim = cfg['model']['hidden_dim']
    device = torch.device("cuda" if device=='cuda' and torch.cuda.is_available() else "cpu")
    model = SAC(state_dim, action_dim, device, hidden_dim)

    ########################
    #    load SAC model    #
    ########################
    model_path = cfg['test']['model_path']
    checkpoint = torch.load(model_path)
    model.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    model.value_net.load_state_dict(checkpoint['value_net_state_dict'])
    model.target_value_net.load_state_dict(checkpoint['value_net_state_dict'])
    model.q_net.load_state_dict(checkpoint['q_net_state_dict'])

    ########################
    #     test settings    #
    ########################
    test_times = cfg['test']['test_times']
    video_path = cfg['test']['video_path']
    env_name = cfg['env']['env_name']

    os.makedirs(video_path, exist_ok=True)
    env = make_env(env_name)

    ########################
    #      test model      #
    ########################
    average_score = 0.0
    seed = [i for i in range(test_times)]
    for idx in range(test_times):
        video_file_path = os.path.join(video_path, f'test_{idx}.mp4')
        video_recorder = VideoRecorder(env, video_file_path)
        state = env.reset(seed=seed[idx])
        done, score = False, 0.0
        while not done:
            with torch.no_grad():
                action = model.get_action(state, validation=True).item()
            next_state, reward, done, info = env.step(action)
            score += reward
            state = next_state
            video_recorder.capture_frame()
        average_score += score
        video_recorder.close()
    average_score /= test_times
    print(f'----- average score: {average_score:6,.2f} -----')