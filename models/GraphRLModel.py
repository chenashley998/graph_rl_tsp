import torch
import numpy as np
import torch.nn.functional as F
import random
from tqdm import tqdm
from models.GAT import GATEncoder
from models.MLPPolicy import PolicyNetwork
from envs.TSP import TSP
from utils.utils import generate_tsp_instance, create_graph_data
import matplotlib.pyplot as plt



class GraphRLModel:
    def __init__(self, config):
        self.model_config = config['model']
        self.training_config = config['training']
        self.env_config = config['env']
        self.evaluation_config = config['evaluation']
        self.general_config = config['general']
        self.masking_config = config['masking']

        self.device = self.general_config['device']
        self.set_seed(self.general_config['seed'])

        self.gat = GATEncoder(
            input_dim=self.model_config['input_dim'],
            hidden_dim=self.model_config['hidden_dim'],
            output_dim=self.model_config['output_dim'],
            num_heads=self.model_config['num_heads']
        ).to(self.device)

        self.policy_net = PolicyNetwork(
            embedding_dim=self.model_config['output_dim'],
            hidden_dim=self.model_config['policy_hidden_dim'],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.gat.parameters()) + list(self.policy_net.parameters()),
            lr=self.training_config['learning_rate']
        )

        self.mask_value = self.masking_config['mask_value']

    def masked_softmax(self, scores, mask):
        # Set scores of visited cities to a large negative value to force 0 probs
        scores = scores.masked_fill(mask, self.mask_value)
        probs = F.softmax(scores, dim=0)
        return probs

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def compute_returns(self, rewards):
        '''
            This function computes the discounted sum of rewards i.e. Gt
        '''
        gamma = self.training_config['gamma']
        eps = self.training_config['eps']
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    def learn(self, resume_model = None):
        num_episodes = self.training_config['num_episodes']
        log_interval = self.training_config['log_interval']
        save_interval = self.training_config['save_interval']
        lr = self.training_config['learning_rate']
        gamma = self.training_config['gamma']
        start_episode = 1
        if resume_model:
            self.load_model(resume_model)
            # Parse the episode this stopped training at to resume from there
            start_episode = int(resume_model.split('_')[4]) + 1 # This assumes specific naming convention: "checkpoints/model_cities_<>_episode_<>..."

        ## With no checkpoint this just trains num_episodes. With checkpoint, it trains an additional num_episodes. 
        for episode in tqdm(range(start_episode, num_episodes + start_episode)):
            # Generate a new TSP instance
            num_cities = self.training_config['num_cities']
            coordinates = generate_tsp_instance(num_cities)
            data = create_graph_data(coordinates).to(self.device)
            env = TSP(coordinates, self.env_config)
            state = env.reset()

            done = False
            log_probs = []
            rewards = []


            while not done:
                embeddings = self.gat(data.x, data.edge_index)
                current_city = state['current_city']
                current_embedding = embeddings[current_city]
                scores = self.policy_net(current_embedding, embeddings)
                visited_mask = torch.tensor(state['visited_mask'], dtype=torch.bool, device=self.device)

                # Compute action probabilities with masking
                action_probs = self.masked_softmax(scores, visited_mask)

                # Sample action from the probability distribution
                m = torch.distributions.Categorical(action_probs)
                action = m.sample()

                next_state, reward, done, _ = env.step(action.item())

                log_probs.append(m.log_prob(action))
                rewards.append(reward)
                state = next_state

            # Compute Gt
            returns = self.compute_returns(rewards)

            # Compute policy loss (this is just REINFORCE i.e. vanilla policy gradient)
            # Currently just doing this for each trajectory which might be unstable??
            ## TODO: Implement mini-batches for more stable training and better convergence
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = torch.stack(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # Logging episode rewards
            ## TODO: Should probably be saving these with a small log interval so I can see reward over time?
            ## TODO: Should probably save these in a better log format for post processing
            if episode % log_interval == 0:
                total_reward = sum(rewards)
                average_reward = total_reward / len(rewards)
                print(f'Episode {episode}\tTotal Reward: {total_reward:.2f}\tAverage Reward per Step: {average_reward:.2f}')
                        
            # Save model checkpoints
            if episode % save_interval == 0:
                ckpt_save_path = f'checkpoints/model_cities_{num_cities}_episode_{episode}_lr_{lr}_gamma_{gamma}'
                ckpt_save_path = ckpt_save_path.replace('.', 'p')
                ckpt_save_path = ckpt_save_path + '.pth'
                self.save_model(ckpt_save_path)
                print(f'Saving model checkpoint at episode {episode} to {ckpt_save_path}')
        
            

    def test(self, checkpoint_path = None):
        num_cities = self.evaluation_config['num_cities']
        num_instances = self.evaluation_config['num_instances']

        ## These are to save results across all instances for some post-processing
        total_rewards = []
        total_lengths = []
        all_instance_coordinates = []
        all_instance_tours = []


        if checkpoint_path:
            self.load_model(checkpoint_path)
            print(f'Testing model loaded from {checkpoint_path}')

        for _ in range(num_instances):
                # Generate a new TSP instance
                coordinates = generate_tsp_instance(num_cities)
                data = create_graph_data(coordinates).to(self.device)
                env = TSP(coordinates, self.env_config)
                state = env.reset()

                done = False
                rewards = []
                tour = [int(state['current_city'])]

                while not done:
                    with torch.no_grad():
                        embeddings = self.gat(data.x, data.edge_index)
                        current_city = state['current_city']
                        current_embedding = embeddings[current_city]
                        scores = self.policy_net(current_embedding, embeddings)
                        visited_mask = torch.tensor(state['visited_mask'], dtype=torch.bool, device=self.device)

                        # Compute action probabilities with masking
                        action_probs = self.masked_softmax(scores, visited_mask)

                        # Select actions greedily
                        action = torch.argmax(action_probs)

                    next_state, reward, done, _ = env.step(action.item())

                    rewards.append(reward)
                    tour.append(int(action.item()))

                    state = next_state

                total_reward = sum(rewards)
                total_rewards.append(total_reward)

                total_length = -total_reward  # Cause rewards are negative distances
                total_lengths.append(total_length)

                all_instance_coordinates.append(coordinates)
                all_instance_tours.append(tour)

                average_length = np.mean(total_lengths)
                print(f'Average Tour Length over {num_instances} instances: {average_length:.2f}')

                return total_rewards, total_lengths, all_instance_coordinates, all_instance_tours

    def save_model(self, path):
        torch.save({
            'gat_state_dict': self.gat.state_dict(),
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.gat.load_state_dict(checkpoint['gat_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])