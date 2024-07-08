import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
def train_model(rank, FLAGS):
    device = xm.xla_device()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        para_loader = pl.ParallelLoader(train_loader, [device])
        for inputs, labels in para_loader.per_device_loader(device):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.to(device))
            loss.backward()
            xm.optimizer_step(optimizer)
            running_loss += loss.item()

            # Calculate training accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels.to(device)).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {train_accuracy:.2f}%')
        xm.mark_step()

        model.eval()
        correct_test = 0
        total_test = 0
        para_loader = pl.ParallelLoader(test_loader, [device])
        with torch.no_grad():
            for inputs, labels in para_loader.per_device_loader(device):
                outputs = model(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                total_test += labels.size(0)
                correct_test += (predicted == labels.to(device)).sum().item()

        test_accuracy = 100 * correct_test / total_test
        print(f'Test Accuracy: {test_accuracy:.2f}%')

FLAGS = {}
xmp.spawn(train_model, args=(FLAGS,), nprocs=1, start_method='fork')
