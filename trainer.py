import torch
from Autoformer.models.Autoformer import Model
from model_encoder import TimeSeriesEncoder
from pretrain_encoder import pretrain_contrastive
from preprocess_dataset import gen_data
import matplotlib.pyplot as plt
from train_contrast_autoformer import train_autoformer, save_model, test_autoformer


class Configs:
    def __init__(self):
        self.seq_len = 8  # Input sequence length (number of past steps)
        self.label_len = 4  # Input label length
        self.pred_len = 1  # Prediction length (number of future steps)
        self.output_attention = False  # Whether to output attention scores
        self.enc_in = 21  # Number of input features for the encoder
        self.dec_in = 21  # Number of input features for the decoder
        self.d_model = 128  # Model dimensionality (embedding size)
        self.n_heads = 8  # Number of attention heads
        self.d_ff = 512  # Dimension of feedforward layers
        self.moving_avg = 25  # Moving average window size
        self.dropout = 0.1  # Dropout rate
        self.activation = 'relu'  # Activation function
        self.e_layers = 4  # Number of encoder layers # before: 2
        self.d_layers = 2  # Number of decoder layers # before: 1
        self.factor = 5  # Factor for the AutoCorrelation layer
        self.c_out = 1  # Output dimension (typically the target dimension)

        # Add the 'embed' attribute
        self.embed = 'timeF'  # Set embedding type (adjust based on the model's expected type)

        # Set 'freq' to represent five-minute intervals
        self.freq = 'h'  # 'T' for minutes, '5' for a 5-minute interval

feature_names, num_features, train_loader, test_loader, sequence_length, scaler = gen_data()

# Initialize encoder and optimizer
# input_dim = traffic_data.shape[1]  # number of features in the dataset
# hidden_dim = 128
# latent_dim = 64

# sequence_length = 6  # Length of the input sequence
input_dim = sequence_length * num_features
# input_dim = num_features  # number of features in the dataset
hidden_dim = 16
latent_dim = sequence_length * num_features

print('input_dim size:', input_dim)

# num_features, train_loader, test_loader,sequence_length, scaler = gen_data()
encoder = TimeSeriesEncoder(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)


# Initialize the configs
configs = Configs()

# Initialize the model with the configs
autoformer = Model(configs)

# Print the model to verify the structure
print(autoformer)

# Initialize Autoformer model with the configs
autoformer = Model(configs)  # Autoformer model initialization

# Initialize the optimizer for the Autoformer model
autoformer_optimizer = torch.optim.Adam(autoformer.parameters(), lr=0.001)

# Train the Autoformer model and get the loss history
loss_history = train_autoformer(autoformer, train_loader, test_loader, encoder, autoformer_optimizer, sequence_length, num_features, scaler, epochs=10)

# Plot the training loss over epochs
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
plt.title("Autoformer Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Save the model
save_model(autoformer, "autoformer_model_1.pth")

# Test the model
test_autoformer(autoformer, test_loader, encoder, sequence_length, num_features, scaler, feature_names)