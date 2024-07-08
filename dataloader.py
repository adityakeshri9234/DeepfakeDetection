import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Define subdirectories and labels
        subdirs = {
            'Celeb-real': 'real',
            'Celeb-synthesis': 'fake',
            'YouTube-real': 'real'
        }

        for subdir, label in subdirs.items():
            label_dir = os.path.join(data_dir, subdir)
            for fname in os.listdir(label_dir):
                if fname.endswith('.npy'):
                    file_path = os.path.join(label_dir, fname)
                    self.samples.append((file_path, label))

        self.label_to_idx = {'real': 0, 'fake': 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        optical_flow = np.load(file_path)
        
        # Ensure the optical flow has 3 channels
        if optical_flow.shape[2] == 2:
            # Add a third channel of zeros
            optical_flow = np.concatenate((optical_flow, np.zeros((optical_flow.shape[0], optical_flow.shape[1], 1))), axis=2)
        elif optical_flow.shape[2] == 1:
            # Duplicate the single channel to make it 3 channels
            optical_flow = np.repeat(optical_flow, 3, axis=2)

        optical_flow = torch.from_numpy(optical_flow).permute(2, 0, 1).float()  # Change to CxHxW and convert to float

        if self.transform:
            optical_flow = self.transform(optical_flow)

        label_idx = self.label_to_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.float32)

        return optical_flow, label_tensor

# Function to create dataloaders
def create_dataloaders(data_dir, batch_size=256, num_workers=4):
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization, modify as needed
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization, modify as needed
    ])

    train_dataset = OpticalFlowDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = OpticalFlowDataset(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

# Example usage
data_dir = 'processed_celeb_df'  # Replace with your dataset path
train_loader, test_loader = create_dataloaders(data_dir)

# Print out a sample to verify
for optical_flow, label in train_loader:
    print(optical_flow.size(), label.size())
    break

