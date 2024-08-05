import torch
from datetime import datetime

Hyper_Param = {
    'today': datetime.now().strftime('%Y-%m-%d'),
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tau': 0.001,
    'discount_factor': 0.9,
    'theta': 0.15,
    'dt': 0.01,
    'sigma': 1.5,
    'epsilon': 1.5,
    'epsilon_decay': 0.9999,
    'epsilon_min': 0.0001,
    'lr_actor': 0.0001,
    'lr_critic': 0.001,
    'batch_size': 512,
    'train_start': 4000,
    'num_episode': 200000,
    'memory_size': 10**5,
    'print_every': 1000,
    'num_neurons': [16,16,32,256],
    'critic_num_neurons': [32,32],
    'step_max': 200,
    'vw_max': 5,
    'window_size': 1000,
    'Saved_using': False,
    'MODEL_PATH': "saved_model",
    'MODEL_NAME': "model_(227, 1001.0).h5"
}

