from models.GraphRLModel import GraphRLModel
from config import config
from utils.utils import *

if __name__ == "__main__":
    model = GraphRLModel(config)

    # This should be a checkpoint path from checkpoints
    # resume_model = 'checkpoints/model_cities_5_episode_1000_lr_0p001_gamma_0p95.pth'
    resume_model = None
    model.learn(resume_model = resume_model)

    # test_checkpoint_path = 'checkpoints/model_cities_5_episode_1000_lr_0p001_gamma_0p95.pth'
    # total_rewards, total_lengths, all_instance_coordinates, all_instance_tours = model.test(checkpoint_path=test_checkpoint_path)

    # Test path visualization
    # coordinates = all_instance_coordinates[0]
    # model_tour = all_instance_tours[0]
    # start_node = model_tour[0]
    # model_tour_length = total_lengths[0]
    # optimal_tour, optimal_tour_length = compute_optimal_tour(coordinates, start_node)
    # compare_tours(coordinates, model_tour, optimal_tour, model_tour_length, optimal_tour_length) 
    # print('dummy')

