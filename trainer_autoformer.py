import torch
from Autoformer.models.Autoformer import Model
from preprocess_dataset import gen_data
import matplotlib.pyplot as plt
from train_autoformer import train_autoformer, save_model, test_autoformer
from datetime import datetime

class Configs:
    def __init__(self):
        self.seq_len = 8  # Input sequence length (number of past steps)
        self.label_len = 4  # Input label length
        self.pred_len = 1  # Prediction length (number of future steps)
        self.output_attention = False
        self.enc_in = 21  # Number of input features for the encoder
        self.dec_in = 21  # Number of input features for the decoder
        self.d_model = 128  # Model dimensionality
        self.n_heads = 8  # Number of attention heads
        self.d_ff = 512  # Feedforward layer dimension
        self.moving_avg = 25
        self.dropout = 0.1  # Dropout rate
        self.activation = 'relu'
        self.e_layers = 2
        self.d_layers = 1
        self.factor = 5
        self.c_out = 1  # Output dimension
        self.embed = 'timeF'
        self.freq = 'h'

# Data loading and preparation
feature_names, num_features, train_loader, test_loader, sequence_length, scaler = gen_data()

# Initialize the model configuration and Autoformer
configs = Configs()
autoformer = Model(configs)

# Initialize optimizer
optimizer = torch.optim.Adam(autoformer.parameters(), lr=0.001)

# Train the Autoformer model and get the loss history
loss_history = train_autoformer(autoformer, train_loader, test_loader, optimizer, sequence_length, num_features, scaler, patience=10, min_delta=1e-4, epochs=50)

# Plot the training loss over epochs
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
plt.title("Autoformer Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
tr_loss_plot_path = "results/" + datetime.now().strftime("%Y%m%d%H%M") + "_training_loss.png"
plt.savefig(tr_loss_plot_path)
print(f"training loss plot saved to {tr_loss_plot_path}")
plt.show()

test_autoformer(autoformer, test_loader, sequence_length, num_features, scaler, feature_names)
