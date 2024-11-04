import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(TimeSeriesEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        # Ensure x has the correct shape: (batch_size, sequence_length, num_features)
        # Flatten if necessary (optional based on your data)
        batch_size, sequence_length, num_features = x.size()

        # Pass through the encoder
        x = x.view(batch_size, sequence_length * num_features)  # Flatten the sequence if needed

        return self.encoder(x)