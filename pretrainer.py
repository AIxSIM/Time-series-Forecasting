import torch
from model_encoder import TimeSeriesEncoder
from pretrain_encoder import pretrain_contrastive
from preprocess_dataset import gen_data
import matplotlib.pyplot as plt


feature_names, num_features, train_loader,sequence_length = gen_data()

input_dim = sequence_length * num_features
hidden_dim = 16
latent_dim = 8

print('input_dim size:', input_dim)

encoder = TimeSeriesEncoder(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

# # Pretrain the encoder using contrastive learning
# pretrain_contrastive(encoder, train_loader, optimizer, epochs=1)

# Train the encoder and get the loss history
loss_history = pretrain_contrastive(encoder, train_loader, optimizer, epochs=10)

# Plot the training loss over epochs
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
plt.title("Contrastive Loss during Encoder Pre-training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()