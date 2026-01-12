### Simple test of training process
from m6_cookiecutter.model import MyAwesomeModel
import torch
import pytest
import os
from m6_cookiecutter.data import corrupt_mnist

@pytest.mark.skipif(not os.path.exists("../../data/processed/train_images.pt"), reason="Processed data not found")
def test_training():
    # Load training dataset
    train_set, _ = corrupt_mnist()
    # Create data loader with batch size of 32
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = MyAwesomeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Set model to training mode
    model.train()
    # Training loop
    for img, target in train_dataloader:
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(img)
        # Calculate loss
        loss = criterion(y_pred, target)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        break  # Just one batch for testing

    # Verify loss is positive
    assert loss.item() > 0



    # tests/test_model.py

from m6_cookiecutter.model import MyAwesomeModel

@pytest.mark.skipif(not os.path.exists("../../data/processed/train_images.pt"), reason="Processed data not found")
def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))

    assert len(train_dataset) == N_train, "Dataset did not have the correct number of samples"





@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    # Initialize the model
    model = MyAwesomeModel()
    # Create random input tensor with shape (batch_size, 1, 28, 28)
    x = torch.randn(batch_size, 1, 28, 28)
    # Forward pass through the model
    y = model(x)
    # Assert output shape is (batch_size, 10) for 10 classes
    assert y.shape == (batch_size, 10)
