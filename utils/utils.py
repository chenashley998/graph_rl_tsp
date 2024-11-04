import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming



def generate_tsp_instance(num_cities):
    coordinates = np.random.rand(num_cities, 2)
    return coordinates

def create_graph_data(coordinates):
    num_nodes = coordinates.shape[0]
    # Create edge index for a fully connected graph
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make edges bidirectional

    # Node features
    x = torch.tensor(coordinates, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data

## TODO: Probably need to re-write this if training and testing in colab or something
def plot_tour(coordinates, tour, total_length, title='TSP Tour', ax = None, show_plot=False):
    coordinates = np.array(coordinates)
    tour = tour + [tour[0]]  # Return to start
    num_cities = coordinates.shape[0]

    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title(f"{title}\nTotal Length: {total_length:.2f}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Plot the nodes
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, zorder=2, label='Cities')

    # Highlight the start node
    start_city = tour[0]
    ax.scatter(coordinates[start_city, 0], coordinates[start_city, 1], c='red', s=150, zorder=3, label='Start City')

    for i in range(len(tour) - 1):
        start_idx = tour[i]
        end_idx = tour[i + 1]
        ax.annotate("",
                    xy=(coordinates[end_idx, 0], coordinates[end_idx, 1]),
                    xytext=(coordinates[start_idx, 0], coordinates[start_idx, 1]),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                    zorder=1)

    ax.legend()
    if show_plot:
        plt.show()

def compute_tour_length(coordinates, tour):
    coordinates = np.array(coordinates)
    tour = tour + [tour[0]]  # Return to start
    total_length = 0.0
    for i in range(len(tour) - 1):
        start_idx = tour[i]
        end_idx = tour[i + 1]
        coord_start = coordinates[start_idx]
        coord_end = coordinates[end_idx]
        distance = np.linalg.norm(coord_start - coord_end)
        total_length += distance
    return total_length

def compute_optimal_tour(coordinates, start_node):
    '''
        This uses DP so might be slow for instances with more than 20 cities
    '''
    num_cities = len(coordinates)
    coordinates = np.array(coordinates)
    
    # Need to get distance matrix from coords first
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix

    optimal_tour, total_length = solve_tsp_dynamic_programming(distance_matrix)
    start_idx = optimal_tour.index(start_node)
    # This tsp solvers can't specify a start node, so just rotate the tour and then add start node to return
    optimal_tour = optimal_tour[start_idx:] + optimal_tour[:start_idx] + [start_node]

    return optimal_tour, total_length

def compare_tours(coordinates, model_tour, optimal_tour, model_length, optimal_length):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_tour(coordinates, model_tour, total_length=model_length, title='Model Tour', ax=axes[0])
    plot_tour(coordinates, optimal_tour, total_length=optimal_length, title='Optimal Tour', ax=axes[1])

    plt.tight_layout()
    plt.show()