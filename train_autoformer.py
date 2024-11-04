import torch.nn.functional as F
import torch

import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def train_autoformer(autoformer, train_loader, test_loader, optimizer, sequence_length, num_features, scaler, patience=5, min_delta=1e-4, epochs=5):
    autoformer.train()  # Set Autoformer to training mode
    loss_history = []  # List to store loss values per epoch
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for data in train_loader:
            x, y, x_mark_enc, x_mark_dec, _ = data  # Extract input tensors and ignore timestamp

            # Prepare inputs for Autoformer
            x = x.view(x.shape[0], sequence_length, num_features)
            x_mark_enc = x_mark_enc.view(x.shape[0], sequence_length, -1)
            x_mark_dec = x_mark_dec.view(x.shape[0], -1, x_mark_enc.shape[-1])

            # Autoformer makes predictions based on sequence data and time markers
            predictions = autoformer(x, x_mark_enc, x, x_mark_dec)

            # Compute the loss between predictions and true future values (y)
            loss = F.mse_loss(predictions, y)

            # Perform backpropagation and update the model
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the Autoformer model's weights

            epoch_loss += loss.item()  # Accumulate the loss for this batch

        # Calculate the average loss for the epoch and store it
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)

        # Print current datetime and loss in the format YY/mm/dd/hh:MM:ss
        current_time = time.strftime("%Y/%m/%d/%H:%M:%S")
        # print(f'[{current_time}] Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}')

        # Validation check for early stopping
        val_loss = evaluate_model(autoformer, test_loader, sequence_length, num_features)
        print(f"[{current_time}] Epoch {epoch + 1}, Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            patience_counter = 0
            # Optionally save the best model so far
            # torch.save(autoformer.state_dict(), "best_autoformer_model.pth")
            save_model(autoformer, "best_autoformer_model")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered @ Epoch {epoch}")
            print(f"[{current_time}] Early Stopping triggered @ Epoch {epoch} (patience:{patience_counter})")
            break

    return loss_history

# Evaluation function for validation loss
def evaluate_model(autoformer, test_loader, sequence_length, num_features):
    autoformer.eval()
    val_loss = 0
    with torch.no_grad():
        for data in test_loader:
            x, y, x_mark_enc, x_mark_dec, _ = data
            x = x.view(x.shape[0], sequence_length, num_features)
            x_mark_enc = x_mark_enc.view(x.shape[0], sequence_length, -1)
            x_mark_dec = x_mark_dec.view(x.shape[0], -1, x_mark_enc.shape[-1])
            predictions = autoformer(x, x_mark_enc, x, x_mark_dec)
            loss = F.mse_loss(predictions, y)
            val_loss += loss.item()

    return val_loss / len(test_loader)

def test_autoformer(autoformer, test_loader, sequence_length, num_features, scaler, feature_names):
    autoformer.eval()  # Set Autoformer to evaluation mode
    actuals, predictions, timestamps = [], [], []
    test_time = datetime.now().strftime("%Y%m%d%H%M")

    with torch.no_grad():
        for data in test_loader:
            x, y, x_mark_enc, x_mark_dec, batch_timestamps = data  # Extract input tensors and timestamps

            # Prepare inputs for Autoformer
            x = x.view(x.shape[0], sequence_length, num_features)
            x_mark_enc = x_mark_enc.view(x.shape[0], sequence_length, -1)
            x_mark_dec = x_mark_dec.view(x.shape[0], -1, x_mark_enc.shape[-1])

            # Autoformer predictions
            preds = autoformer(x, x_mark_enc, x, x_mark_dec)
            actual = y.cpu().numpy()

            actuals.append(actual)
            predictions.append(preds.cpu().numpy())

            # Convert timestamp tensor back to numpy.datetime64 for plotting
            timestamps_batch = pd.to_datetime(batch_timestamps.cpu().numpy().flatten(), unit='s')
            timestamps.extend(timestamps_batch)

    # Convert lists to arrays
    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Flatten arrays to 2D for RMSE calculation
    actuals_reshaped = actuals.reshape(-1, actuals.shape[-1])
    predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])

    # Calculate RMSE by feature in scaled space
    rmse_scaled_by_feature = np.sqrt(
        mean_squared_error(actuals_reshaped, predictions_reshaped, multioutput='raw_values'))

    # Inverse transform to original scale and calculate RMSE by feature in original space
    actuals_orig = scaler.inverse_transform(actuals_reshaped)
    predictions_orig = scaler.inverse_transform(predictions_reshaped)
    rmse_orig_by_feature = np.sqrt(mean_squared_error(actuals_orig, predictions_orig, multioutput='raw_values'))

    # **Extract RMSE for TR_VOL_5MIN (assuming it is the second feature)**
    tr_vol_5min_index = feature_names.index('TR_VOL_5MN')
    rmse_tr_vol_5min_scaled = rmse_scaled_by_feature[tr_vol_5min_index]
    rmse_tr_vol_5min_orig = rmse_orig_by_feature[tr_vol_5min_index]

    # **Print RMSE values to console**
    print(f"RMSE for TR_VOL_5MIN (Scaled): {rmse_tr_vol_5min_scaled:.4f}")
    print(f"RMSE for TR_VOL_5MIN (Original Scale): {rmse_tr_vol_5min_orig:.4f}")

    # **Write RMSE values to a file**
    output_path = "results/"+test_time+"_RMSE_TR_VOL_5MIN.csv"
    with open(output_path, "w") as f:
        f.write(f"TR_VOL_5MIN (Scaled), TR_VOL_5MIN (Original Scale)\n")
        f.write(f"{rmse_tr_vol_5min_scaled:.4f}, {rmse_tr_vol_5min_orig:.4f}\n")
    print(f"RMSE results saved to {output_path}")

    # Plotting RMSE by feature (scaled and original) in bar graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].bar(feature_names, rmse_scaled_by_feature, color='skyblue')
    axes[0].set_title("RMSE by Feature (Scaled)")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis='x', rotation=90)

    axes[1].bar(feature_names, rmse_orig_by_feature, color='salmon')
    axes[1].set_title("RMSE by Feature (Original Scale)")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis='x', rotation=90)

    # Save RMSE bar plot as PNG
    plt.tight_layout()
    rmse_bar_plot_path = "results/"+test_time+"_RMSE_plot_by_feature_TR_VOL_5MIN.png"
    plt.savefig(rmse_bar_plot_path)
    print(f"RMSE bar plot saved to {rmse_bar_plot_path}")
    plt.show()

    # Extract and plot TR_VOL_5MIN feature for scaled and original values
    timestamps = pd.to_datetime(timestamps)  # Use timestamps directly for x-axis

    # Extract actuals and predictions for the TR_VOL_5MIN feature (assuming it's the second feature)
    actuals_tr_vol_5min_scaled = actuals_reshaped[:, 1]
    predictions_tr_vol_5min_scaled = predictions_reshaped[:, 1]
    actuals_tr_vol_5min_orig = actuals_orig[:, 1]
    predictions_tr_vol_5min_orig = predictions_orig[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(timestamps, actuals_tr_vol_5min_scaled, label='Actual (Scaled)', color='blue')
    axes[0].plot(timestamps, predictions_tr_vol_5min_scaled, label='Prediction (Scaled)', color='orange',
                 linestyle='--')
    axes[0].set_title("TR_VOL_5MIN Comparison (Scaled)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Scaled Value")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].plot(timestamps, actuals_tr_vol_5min_orig, label='Actual (Original Scale)', color='blue')
    axes[1].plot(timestamps, predictions_tr_vol_5min_orig, label='Prediction (Original Scale)', color='orange',
                 linestyle='--')
    axes[1].set_title("TR_VOL_5MIN Comparison (Original Scale)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Original Value")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    # Save time series comparison plots as PNGs
    plt.tight_layout()
    tr_vol_scaled_plot_path = "results/"+test_time+"_RMSE_plot_comparison_scaled_TR_VOL_5MIN.png"
    tr_vol_orig_plot_path = "results/"+test_time+"_RMSE_plot_comparison_original_TR_VOL_5MIN.png"
    fig.savefig(tr_vol_scaled_plot_path)
    print(f"TR_VOL_5MIN Comparison (Scaled) plot saved to {tr_vol_scaled_plot_path}")
    fig.savefig(tr_vol_orig_plot_path)
    print(f"TR_VOL_5MIN Comparison (Original Scale) plot saved to {tr_vol_orig_plot_path}")
    plt.show()

    return rmse_scaled_by_feature, rmse_orig_by_feature


# Save the trained model
def save_model(model, path="autoformer_model"):
    path = "models/"+path+"_"+datetime.now().strftime("%Y%m%d%H%M")+".pth"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load the trained model
def load_model(model, path="autoformer_model.pth"):
    path = "models/"+path
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
