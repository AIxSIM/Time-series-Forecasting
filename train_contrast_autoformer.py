import torch.nn.functional as F
import torch

# Define the training loop for Autoformer
def train_autoformer(autoformer, train_loader, test_loader, encoder, optimizer, sequence_length, num_features, scaler,
                     epochs=10):
    autoformer.train()  # Set Autoformer to training mode
    loss_history = []  # List to store loss values per epoch

    for epoch in range(epochs):
        epoch_loss = 0
        for data in train_loader:
            # Extract only the required tensors for training
            x, y, x_mark_enc, x_mark_dec, _ = data  # Use underscore to ignore timestamp

            # Preprocess the sequence data using the pre-trained encoder
            with torch.no_grad():  # Encoder is pre-trained, so no gradients are needed
                z = encoder(data[0])  # Extract latent representation from encoder (data[0] is the sequence)
            z = z.view(z.shape[0], sequence_length, num_features)

            # Extract the time markers for the encoder and decoder
            x_mark_enc = data[2]  # Time markers for encoder input
            x_mark_dec = data[3]  # Time markers for decoder input

            # Autoformer makes predictions based on the encoder's latent representation
            predictions = autoformer(z, x_mark_enc, z, x_mark_dec)

            # Compute the loss between predictions and true future values (data[1] is the target)
            loss = F.mse_loss(predictions, data[1])

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
        print(f'[{current_time}] Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}')

    return loss_history

# Save the trained model
def save_model(model, path="autoformer_model.pth"):
    path = "models/"+path
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load the trained model
def load_model(model, path="autoformer_model.pth"):
    path = "models/"+path
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")


def test_autoformer(autoformer, test_loader, encoder, sequence_length, num_features, scaler, feature_names):
    autoformer.eval()  # Set Autoformer to evaluation mode
    actuals, predictions, timestamps = [], [], []

    with torch.no_grad():
        for data in test_loader:
            z = encoder(data[0])  # Extract latent representation
            z = z.view(z.shape[0], sequence_length, num_features)
            x_mark_enc = data[2]
            x_mark_dec = data[3]
            batch_timestamps = data[4]  # Access timestamps

            preds = autoformer(z, x_mark_enc, z, x_mark_dec)
            actual = data[1].cpu().numpy()

            actuals.append(actual)
            predictions.append(preds.cpu().numpy())

            # Convert timestamp tensor back to numpy.datetime64
            timestamps_batch = pd.to_datetime(batch_timestamps.cpu().numpy().flatten(), unit='s')
            timestamps.extend(timestamps_batch)

    # Convert lists to arrays
    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Flatten arrays to 2D for RMSE calculation
    actuals_reshaped = actuals.reshape(-1, actuals.shape[-1])
    predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])

    # RMSE by feature in scaled and original scales
    rmse_scaled_by_feature = np.sqrt(
        mean_squared_error(actuals_reshaped, predictions_reshaped, multioutput='raw_values'))
    actuals_orig = scaler.inverse_transform(actuals_reshaped)
    predictions_orig = scaler.inverse_transform(predictions_reshaped)
    rmse_orig_by_feature = np.sqrt(mean_squared_error(actuals_orig, predictions_orig, multioutput='raw_values'))

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

    plt.tight_layout()
    plt.show()

    # # Flatten timestamps and convert to datetime
    # timestamps = np.concatenate(timestamps).flatten()

    # Extract actuals and predictions for the TR_VOL_5MIN feature (assuming it's the second feature)
    actuals_tr_vol_5min_scaled = actuals_reshaped[:, 1]
    predictions_tr_vol_5min_scaled = predictions_reshaped[:, 1]
    actuals_tr_vol_5min_orig = actuals_orig[:, 1]
    predictions_tr_vol_5min_orig = predictions_orig[:, 1]

    # Plot scaled and original comparisons
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scaled comparison
    axes[0].plot(timestamps, actuals_tr_vol_5min_scaled, label='Actual (Scaled)', color='blue')
    axes[0].plot(timestamps, predictions_tr_vol_5min_scaled, label='Prediction (Scaled)', color='orange',
                 linestyle='--')
    axes[0].set_title("TR_VOL_5MIN Comparison (Scaled)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Scaled Value")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Original scale comparison
    axes[1].plot(timestamps, actuals_tr_vol_5min_orig, label='Actual (Original Scale)', color='blue')
    axes[1].plot(timestamps, predictions_tr_vol_5min_orig, label='Prediction (Original Scale)', color='orange',
                 linestyle='--')
    axes[1].set_title("TR_VOL_5MIN Comparison (Original Scale)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Original Value")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return rmse_scaled_by_feature, rmse_orig_by_feature
