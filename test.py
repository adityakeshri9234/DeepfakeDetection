train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization, modify as needed
    ])
    
test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization, modify as needed
    ])
train_dataset = OpticalFlowDataset(os.path.join(data_dir, 'train'), transform=train_transform)
test_dataset = OpticalFlowDataset(os.path.join(data_dir, 'test'), transform=test_transform)
dataset_loader=DataLoader(train_dataset+test_dataset, batch_size=256, shuffle=True, num_workers=128)
device = xm.xla_device()
model.eval()
correct_test = 0
total_test = 0
para_loader = pl.ParallelLoader(dataset_loader, [device])
with torch.no_grad():
    for inputs, labels in para_loader.per_device_loader(device):
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        total_test += labels.size(0)
        correct_test += (predicted == labels.to(device)).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f'Test Accuracy: {test_accuracy:.2f}%')
