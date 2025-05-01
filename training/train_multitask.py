# training/train_multitask.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from models.multitask_model import MultiTaskModel

def fake_data(batch_size=4):
    """Generate fake input sentences and random labels for each task
    this is to simulate a batch for tesing the training loop
    """
    # sample personal sentences that i made personally
    sentences = [
        "Hello, how are you?",
        "Purdue University is a great place to study Computer Science.",
        "Fetch Rewards is Great",
        "My name is Anshul."
    ] * (batch_size // 4) # repeat to fill the batch size

    # create random labels for each task
    # task A: 3 classes (0, 1, 2) (greeting, university, company)
    # task B: 2 classes (0, 1) (sentiment: positive, negative)
    labels_a = torch.randint(0, 3, (batch_size,))
    labels_b = torch.randint(0, 2, (batch_size,))
    return sentences, labels_a, labels_b

def train_step(model, optimizer, loss_fn_a, loss_fn_b):
    """
    simulate a single training step:
    ...get a fake batch
    -run forward pass
    - calculate both losses
    - backpropogate and update weights
    - return the loss values
    """
    model.train() # enable training mode (dropout, batch norm, etc.)

    # get a fake batch
    sentences, labels_a, labels_b = fake_data()

    # reset gradients
    optimizer.zero_grad()

    # forward pass through both task heads
    logits_a, logits_b = model(sentences)

    # Calculate individual losses for each task
    loss_a = loss_fn_a(logits_a, labels_a)
    loss_b = loss_fn_b(logits_b, labels_b)

    # Combine losses (equal weighting) (simple sum)
    loss = loss_a + loss_b
    # backpropogate
    loss.backward()
    # update weights
    optimizer.step()

    return loss.item(), loss_a.item(), loss_b.item()

def run_training_loop(epochs=3):
    """
    run a simple multitask training loop fpor a few epochs using fake data
    this demonstrates the training process and how loss is computed and combined across multiple tasks
    """

    # initialize the model, optimizer, and loss functions
    model = MultiTaskModel(num_classes_a=3, num_classes_b=2)
    # use adam optimizer
    # adam optimizer is a popular choice for training transformer models
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    # use cross entropy loss for both tasks
    loss_fn_a = nn.CrossEntropyLoss()
    loss_fn_b = nn.CrossEntropyLoss()

    print("Starting dummy training loop...\n")
    for epoch in range(epochs):
        total_loss, loss_a, loss_b = train_step(model, optimizer, loss_fn_a, loss_fn_b)
        print(f"Epoch {epoch+1} | Total Loss: {total_loss:.4f} | Task A Loss: {loss_a:.4f} | Task B Loss: {loss_b:.4f}")

if __name__ == "__main__":
    run_training_loop()
