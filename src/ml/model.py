import torch
import torch.nn as nn

class AutoencoderModel(nn.Module):
    """Autoencoder neural network for dimensionality reduction and reconstruction."""

    def __init__(self, input_dim: int = 48, encoding_dim: int = 2, hidden_dim: int = 8):
        """
        Initialize the autoencoder model.

        Args:
            input_dim: Dimension of input data
            encoding_dim: Dimension of the encoded representation
            hidden_dim: Dimension of the hidden layer
        """
        super(AutoencoderModel, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input data to the latent space."""
        return self.encoder(x)