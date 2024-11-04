import torch

model_config = {
    'input_dim': 2,
    'hidden_dim': 32,
    'output_dim': 32,
    'num_heads': 4,
    'policy_hidden_dim': 128
}

training_config = {
    'learning_rate': 1e-3,
    'num_episodes': 1000, # This is how many episodes you want this model to train right now (not the total up to which you train).
    'log_interval': 100,
    'save_interval': 1000,
    'gamma': 0.95,
    'eps': 1e-9,
    'entropy_beta': 0.01,
    'num_cities': 5
}

env_config = {
    'invalid_action_penalty': -10.0,
    'return_to_start': True
}

evaluation_config = {
    'num_cities': 5,
    'num_instances': 1
}

general_config = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

masking_config = {
    'mask_value': -1e9
}

config = {
    'model': model_config,
    'training': training_config,
    'env': env_config,
    'evaluation': evaluation_config,
    'general': general_config,
    'masking': masking_config
}
