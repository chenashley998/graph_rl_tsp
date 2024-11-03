import gym
from gym import spaces
import numpy as np



class TSP(gym.Env):
    def __init__(self, coordinates, env_config):
        super(TSP, self).__init__()
        self.coordinates = coordinates
        self.num_cities = coordinates.shape[0]
        self.action_space = spaces.Discrete(self.num_cities)
        self.observation_space = spaces.Dict({
            'current_city': spaces.Discrete(self.num_cities),
            'visited_mask': spaces.MultiBinary(self.num_cities)
        })
        self.invalid_action_penalty = env_config['invalid_action_penalty']
        self.return_to_start = env_config['return_to_start']
        self.reset()

    def _get_observation(self):
        return {
          'current_city': self.current_city,
          'visited_mask': self.visited_mask.copy()
        }

    def reset(self):
        self.current_city = np.random.randint(self.num_cities)
        self.visited_mask = np.zeros(self.num_cities, dtype=bool)
        self.visited_mask[self.current_city] = True
        self.tour = [self.current_city]
        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action)
        if self.visited_mask[action]:
            # Invalid action, penalize heavily
            reward = self.invalid_action_penalty
            done = True
        else:
            # Compute distance traveled
            coord_current = self.coordinates[self.current_city]
            coord_next = self.coordinates[action]
            distance = np.linalg.norm(coord_current - coord_next)
            reward = -distance
            self.current_city = action
            self.visited_mask[action] = True
            self.tour.append(action)
            done = self.visited_mask.all()
            ## This is just manually making sure you will go back to the start city after you've seen all the cities
            if done and self.return_to_start:
                coord_last = self.coordinates[self.current_city]
                coord_start = self.coordinates[self.tour[0]]
                distance = np.linalg.norm(coord_last - coord_start)
                reward += -distance
                self.tour.append(self.tour[0])
        return self._get_observation(), reward, done, {}
