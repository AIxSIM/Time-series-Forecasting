import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class TrafficDataset(Dataset):
    def __init__(self, data, time_features, timestamps, sequence_length=24, target_length=24):
        """
        Args:
            data: Pandas dataframe containing traffic volume and other features.
            sequence_length: Number of past time steps used as input.
            target_length: Number of future time steps to predict.
        """
        self.data = data
        # self.data = data.drop(columns=['time'])
        self.time_features = time_features  # Time markers or timestamps corresponding to the data
        # self.timestamps = timestamps  # Store timestamps
        self.timestamps = (timestamps - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

        self.sequence_length = sequence_length
        self.target_length = target_length

        # Normalize the data
        self.scaler = MinMaxScaler()
        # self.scaler = scaler
        print(data)
        self.data_scaled = self.scaler.fit_transform(data)

    def __len__(self):
        return len(self.data_scaled) - self.sequence_length - self.target_length

    # def __getitem__(self, idx):
    #     # Get input sequence and target sequence
    #     x = self.data_scaled[idx:idx + self.sequence_length]
    #     y = self.data_scaled[idx + self.sequence_length: idx + self.sequence_length + self.target_length]
    #     return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        # # Get the input sequence (x) and the target sequence (y)
        # x = self.data[idx:idx + self.sequence_length]
        # y = self.data[idx + self.sequence_length: idx + self.sequence_length + self.target_length]
        #
        # # Get the time markers for both encoder (x) and decoder (y)
        # x_mark_enc = self.time_features[idx:idx + self.sequence_length]
        # x_mark_dec = self.time_features[idx + self.sequence_length: idx + self.sequence_length + self.target_length]
        # Ensure that x and y are numpy arrays
        x = self.data_scaled[idx:idx + self.sequence_length].values if isinstance(self.data_scaled, pd.DataFrame) else self.data_scaled[idx:idx + self.sequence_length]
        y = self.data_scaled[idx + self.sequence_length: idx + self.sequence_length + self.target_length].values if isinstance(self.data_scaled, pd.DataFrame) else self.data_scaled[idx + self.sequence_length: idx + self.sequence_length + self.target_length]

        # Ensure time_features are numpy arrays
        x_mark_enc = self.time_features[idx:idx + self.sequence_length].values if isinstance(self.time_features, pd.DataFrame) else self.time_features[idx:idx + self.sequence_length]
        x_mark_dec = self.time_features[idx + self.sequence_length: idx + self.sequence_length + self.target_length].values if isinstance(self.time_features, pd.DataFrame) else self.time_features[idx + self.sequence_length: idx + self.sequence_length + self.target_length]

        # scaler =  self.scaler
        timestamp = self.timestamps[idx + self.sequence_length: idx + self.sequence_length + self.target_length]

        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(x_mark_enc, dtype=torch.float32),
                torch.tensor(x_mark_dec, dtype=torch.float32),
                torch.tensor(timestamp, dtype=torch.float32))


# Train-test split function
def split_data(dataset, test_size=0.2):
    train_data, test_data = train_test_split(dataset, test_size=test_size, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, test_loader

# MRT_TF_INFO_1860217400.csv
# def gen_data(file_path='datasets/MRT_TF_INFO_2021M07_1830000302.csv', test_size=0.2, batch_size=32):
def gen_data(file_path='datasets/MRT_TF_INFO_1860217400.csv', test_size=0.2, batch_size=32):
    # Load and prepare data
    traffic_data = pd.read_csv(file_path).drop(columns=['LINK_ID', 'ROAD_ROUTE_ID']).sort_values('datetime')
    traffic_data = traffic_data.fillna(0)

    # Convert datetime to pandas datetime format and create time features
    # traffic_data['datetime'] = pd.to_datetime(traffic_data['datetime'], format='%Y%m%d%H%M')
    traffic_data['datetime'] = pd.to_datetime(traffic_data['datetime'], format='%Y-%m-%d %H:%M:%S')

    timestamps = traffic_data['datetime'].values  # Store datetime for plotting

    traffic_data['hour'] = traffic_data['datetime'].dt.hour
    traffic_data['day_of_week'] = traffic_data['datetime'].dt.dayofweek
    traffic_data['month'] = traffic_data['datetime'].dt.month
    traffic_data['is_weekend'] = traffic_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Cyclical time features
    traffic_data['hour_sin'] = np.sin(2 * np.pi * traffic_data['hour'] / 24)
    traffic_data['hour_cos'] = np.cos(2 * np.pi * traffic_data['hour'] / 24)
    traffic_data['day_of_week_sin'] = np.sin(2 * np.pi * traffic_data['day_of_week'] / 7)
    traffic_data['day_of_week_cos'] = np.cos(2 * np.pi * traffic_data['day_of_week'] / 7)

    # Extract feature names
    feature_names = traffic_data.drop(columns=['datetime']).columns.tolist()

    # Drop datetime and set up feature arrays
    time_features = traffic_data[['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']].values
    data = traffic_data.drop(columns=['datetime']).values

    # Scaling data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # # Split data into train and test
    # train_data, test_data, train_time_features, test_time_features = train_test_split(
    #     data_scaled, time_features, test_size=test_size, shuffle=False
    # )

    # Split data and timestamps into train and test sets
    train_data, test_data, train_time_features, test_time_features, train_timestamps, test_timestamps = train_test_split(
        data_scaled, time_features, timestamps, test_size=test_size, shuffle=False
    )

    # Define sequence and target lengths for the dataset
    sequence_length = 8  # For example, 8 time steps as input
    target_length = 1  # Predict the next time step

    # Create TrafficDataset instances for train and test
    train_dataset = TrafficDataset(train_data, train_time_features, train_timestamps, sequence_length, target_length)
    test_dataset = TrafficDataset(test_data, test_time_features, test_timestamps, sequence_length, target_length)

    print(f"Total samples in train dataset: {len(train_dataset)}")
    print(f"Total samples in test dataset: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return feature_names, data.shape[1], train_loader, test_loader, sequence_length, scaler

# def gen_data():
#     # Load your data (replace with your actual traffic data file)
#     traffic_data = pd.read_csv('datasets/MRT_TF_INFO_2021M07_1830000302.csv').drop(columns=['LINK_ID','ROAD_ROUTE_ID']).sort_values('datetime')
#
#     # Assuming 'time' column is in 'YYYYMMDDHHmm' format as string
#     traffic_data = traffic_data.fillna(0)
#
#     traffic_data['datetime'] = pd.to_datetime(traffic_data['datetime'], format='%Y%m%d%H%M')
#
#     # Extract time features
#     traffic_data['hour'] = traffic_data['datetime'].dt.hour
#     traffic_data['day_of_week'] = traffic_data['datetime'].dt.dayofweek  # Monday=0, Sunday=6
#     traffic_data['month'] = traffic_data['datetime'].dt.month
#
#     # Optional: Create a binary weekend feature
#     traffic_data['is_weekend'] = traffic_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
#
#     # Create cyclical features for time-based attributes
#     traffic_data['hour_sin'] = np.sin(2 * np.pi * traffic_data['hour'] / 24)
#     traffic_data['hour_cos'] = np.cos(2 * np.pi * traffic_data['hour'] / 24)
#
#     traffic_data['day_of_week_sin'] = np.sin(2 * np.pi * traffic_data['day_of_week'] / 7)
#     traffic_data['day_of_week_cos'] = np.cos(2 * np.pi * traffic_data['day_of_week'] / 7)
#
#     # Create the time_features array
#     time_features = traffic_data[['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']]
#
#     print(traffic_data.columns)
#     print(traffic_data[0:5])
#
#     # Create dataset
#     sequence_length = 8  # last 24 hours of traffic data as input
#     target_length = 1  # predict the next 24 hours
#
#     # scaler = MinMaxScaler()
#     traffic_dataset = TrafficDataset(traffic_data.drop(columns=['datetime']), time_features, sequence_length, target_length)
#     print('TD: ', traffic_dataset)
#
#     # # Create data loader for batching
#     # data_loader = DataLoader(traffic_dataset, batch_size=32, shuffle=True)
#
#     # Create DataLoader for batching
#     batch_size = 32
#
#     train_data, test_data = train_test_split(traffic_dataset, test_size=0.2)
#
#     # train_data, test_data = train_test_split(traffic_data.drop(columns=['datetime']), test_size=0.2)
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
#     # train_loader = DataLoader(traffic_dataset, batch_size=batch_size, shuffle=True)
#
#     return traffic_data.drop(columns=['datetime']).shape[1], train_loader, test_loader, sequence_length, traffic_dataset.scaler

