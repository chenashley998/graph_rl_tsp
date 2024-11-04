import torch
import torch.nn.functional as F


class PolicyNetwork(torch.nn.Module):
    '''
        Currently this is a very simple feedforward network of 2 linear layers. 
        The first one takes the current embedding and concatenates it with every existing embedding before passing it through linear layer.
        (Could re-think how we combine information about the current embedding with other embeddings in the score function)
        Then we just take this output (after non-linearity) and pass it through a final linear layer to get scores. 
    '''
    def __init__(self, embedding_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # Output a scalar score

    def forward(self, current_embedding, embeddings):
        num_nodes = embeddings.size(0)
        # Repeat current embedding to match embeddings size
        current_embedding_expanded = current_embedding.unsqueeze(0).repeat(num_nodes, 1)
        # Concatenate embeddings
        combined = torch.cat((current_embedding_expanded, embeddings), dim=1)
        x = F.relu(self.fc1(combined))
        scores = self.fc2(x).squeeze()  # Shape: (num_nodes,)
        return scores