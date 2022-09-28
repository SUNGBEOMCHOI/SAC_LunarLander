## Requirements

### Conda virtual environment setup

```
conda create -n "environment name" python=3.8
conda activate "environment name"
```

### Requirements

```
pip install numpy==1.23
pip install pytorch==1.12.1
pip install gym==0.22
```

### Git clone repo

```
git clone https://github.com/SUNGBEOMCHOI/SAC_LunarLander.git
```

## LunarLander-v2

The Lunar Lander environment is a task that scores high when you fly a spacecraft in the air and land accurately on the landing site.

![https://miro.medium.com/max/720/1*i7lxpgt2K3Q8lgEPJu3_xA.png](https://miro.medium.com/max/720/1*i7lxpgt2K3Q8lgEPJu3_xA.png)

### State

It has the form of an 8-dimensional vector.

- Lander’s x, y position [float]
- Lander’s x, y velocity [float]
- Lander's angle [float]
- Lander's angle velocity [float]
- Whether Lander's left leg touches the ground or not [boolean]
- Whether Lander's right leg touches the ground or not [boolean]

### Reward

The following is the case of returning a high reward.

- How close to the landing pad
- How slowly Lander moves
- How well Lander balanced
- Each of Lander's feet on the ground : +10
- Frame that fires the Lander's side engine: -0.03
- Frame that ignites main engine: -0.3
- Lander breaks: -100
- Lander lands on the right spot: +100

### Action

**There are four actions Lander can do.**

- No-op
- Ignite left engine
- Ignite main engine
- Ignite right engine

## Train

### Config file

You can change ./config/config.yaml file if you want. Current file optimizes to solve LunarLander-v2 task. If you want to change task, you can change env_name, state_dim and action_dim. For example, env_name:CartPole-v1, state_dim:4, action_dim:2

```
device: cuda
train:
  batch_size: 256
  lr: 0.0005
  train_timesteps: 100000
  val_timesteps: 1000
  target_update_timesteps: 1000
  target_update_ratio: 0.05
  discount_factor: 0.99
  replay_buffer_size: 5000
  train_starts_timesteps : 1000
  video_path: ./video
  model_path: ./pretrained
  progress_path: ./train_progress
test:
  test_times: 5
  video_path: ./video
  model_path: ./pretrained/model_75000.pt
env:
  env_name: LunarLander-v2
  state_dim: 8
  action_dim: 4
model:
  hidden_dim: 256
```

### Train code

Model parameters, playing videos, and learning progress graphs are stored every 1000 time steps.

```
python train.py
```

### Train result

Training for 100,000 time steps. The score gradually increases as learning progresses.

![https://github.com/SUNGBEOMCHOI/SAC_LunarLander/blob/main/train_progress/step_100000.png?raw=true](https://github.com/SUNGBEOMCHOI/SAC_LunarLander/blob/main/train_progress/step_100000.png?raw=true)

After training 100,000 timesteps

![https://github.com/SUNGBEOMCHOI/SAC_LunarLander/blob/main/video/LunarLander-v2.gif?raw=true](https://github.com/SUNGBEOMCHOI/SAC_LunarLander/blob/main/video/LunarLander-v2.gif?raw=true)

### Test code

The test code stores the play video and outputs an average score. You have to change model path of config.yaml file before proceeding the test code

```
python test.py
```