import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmaReparam(nn.Module):
    """
    This class reparametrizes the weight of a linear layer using spectral normalization 
    and an additional learned scalar. The goal is to stabilize the training of Transformers 
    by preventing attention entropy collapse.
    """
    
    def __init__(self, out_features, in_features):
        """
        Initializes the SigmaReparam module.
        
        Args:
        - out_features (int): Size of the output dimension.
        - in_features (int): Size of the input dimension.
        """
        
        super(SigmaReparam, self).__init__()
        
        # Initialize gamma (learned scalar) to 1
        self.gamma = nn.Parameter(torch.ones(1))
        
        # Initialize a linear layer with spectral normalization
        self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features, bias=False))

    def forward(self, x):
        """
        Forward pass for SigmaReparam module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Reparametrized output tensor.
        """
        
        # Compute the spectral norm (largest singular value) of the weight matrix
        sigma = torch.linalg.norm(self.linear.weight, ord=2)
        
        # Reparametrize the weight matrix
        W_reparam = self.gamma / sigma * self.linear.weight
        
        return F.linear(x, W_reparam)
