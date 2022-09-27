import os
from collections import namedtuple

import yaml
import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from env import make_env
from model import SAC
from utils import ReplayBuffer, loss_func, optim_func, scheduler_func, plot_progress, update_params

def train(cfg):
    ########################
    #    make SAC model    #
    ########################
    device = cfg['device']
    state_dim = cfg['env']['state_dim']
    action_dim = cfg['env']['action_dim']
    hidden_dim = cfg['model']['hidden_dim']
    model = SAC(state_dim, action_dim, device, hidden_dim)

    ########################
    #    train settings    #
    ########################
    batch_size = cfg['train']['batch_size']
    learning_rate = cfg['train']['lr']
    train_timesteps = cfg['train']['train_timesteps']
    val_timesteps = cfg['train']['val_timesteps']
    target_update_timesteps = cfg['train']['target_update_timesteps']
    discount_factor = cfg['train']['discount_factor']
    replay_buffer_size = cfg['train']['replay_buffer_size']
    train_starts_timesteps = cfg['train']['train_starts_timesteps']
    video_path = cfg['train']['video_path']
    model_path = cfg['train']['model_path']
    progress_path = cfg['train']['progress_path']
    env_name = cfg['env']['env_name']

    os.makedirs(video_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    env = make_env(env_name)
    val_env = make_env(env_name)

    Sample = namedtuple("Data", 'state, action, reward, next_state, done')
    replay_buffer = ReplayBuffer(buf_size=replay_buffer_size, device=device)

    criterion_list = loss_func()
    optim_list = optim_func(model, learning_rate)
    scheduler_list = scheduler_func(optim_list)

    history = {'score':[], 'loss':[0.0]*train_starts_timesteps}
    
    ########################
    #      train model     #
    ########################
    step = 0
    while step <= train_timesteps:
        state = env.reset()
        done, total_reward = False, 0.0
        while not done:
            step += 1
            action = model.get_action(state).item()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            replay_buffer.append(Sample(state, 
                                        np.array([action]),
                                        np.array([reward]),
                                        next_state,
                                        np.array([done])))
            state = next_state
            if len(replay_buffer.memory) > train_starts_timesteps:
                loss = update_params(replay_buffer, model, criterion_list, optim_list, discount_factor, batch_size)
                history['loss'].append(loss)
                for scheduler in scheduler_list:
                    scheduler.step()

            if step % target_update_timesteps == 0:
                model.target_value_net.load_state_dict(model.value_net.state_dict())

            if step % val_timesteps == 0:
                plot_progress(history)
                validation(model, val_env, step, video_path)

        new_steps = step - len(history['score'])
        history['score'].extend([total_reward]*new_steps)

def validation(model, val_env, step, video_path):
    """
    validation test for 5 episodes. 
    Print average scores, Save playing video, Save model

    Args:
        model: SAC model
        val_env: Gym environment for validation
        step: Current time step
        seed: Random seed number
    """
    test_num, average_score = 1, 0.0
    seed = [i for i in range(test_num)]
    video_file_path = os.path.join(video_path, f'step_{step}.mp4')
    video_recorder = VideoRecorder(val_env, video_file_path)
    for idx in range(test_num):
        state = val_env.reset(seed=seed[idx])
        done, score = False, 0.0
        while not done:
            with torch.no_grad():
                action = model.get_action(state).item()
            next_state, reward, done, info = val_env.step(action)
            score += reward
            state = next_state
            val_env.render()
            if (idx+1) == test_num:
                video_recorder.capture_frame()
        average_score += score
    video_recorder.close()
    average_score /= test_num
    print(f'----- train steps: {step:07d} ----- average score: {average_score:6,.2f} -----')

    torch.save({
        'policy_net_state_dict': model.policy_net.state_dict(),
        'value_net_state_dict': model.value_net.state_dict(),
        'q_net_state_dict': model.q_net.state_dict(),
        }, f'./pretrained/model_{step}.pt')

if __name__ == '__main__':
    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    train(cfg)