from models.GraphRLModel import GraphRLModel
from config import config

if __name__ == "__main__":
    model = GraphRLModel(config)

    # This should be a checkpoint path from checkpoints
    # resume_model = 'checkpoints/model_cities_5_episode_1000_lr_0p001_gamma_0p95.pth'
    # model.learn(resume_model = resume_model)

    test_checkpoint_path = 'checkpoints/model_cities_5_episode_1000_lr_0p001_gamma_0p95.pth'
    total_rewards, total_lengths, all_instance_coordinates, all_instance_tours = model.test(checkpoint_path=test_checkpoint_path)
