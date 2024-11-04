from loss import contrastive_loss
from data_aug import augment_data

def pretrain_contrastive(encoder, data_loader, optimizer, epochs=10):
    encoder.train()
    loss_history = []  # List to store loss for each epoch

    for epoch in range(epochs):
        epoch_loss = 0
        for data in data_loader:
            # Apply augmentation to create positive pairs
            augmented_data = augment_data(data)
            # print('data shape:',data[0].shape)
            # print('aug_data shape:',augmented_data[0].shape)
            # Pass both original and augmented data through encoder
            z_i = encoder(data[0])
            z_j = encoder(augmented_data[0])

            # Compute contrastive loss
            loss = contrastive_loss(z_i, z_j)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)

        # print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}')

        print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')

    return loss_history