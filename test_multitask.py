# test_multitask.py

from models.multitask_model import MultiTaskModel
import torch

def test_multitask_model():
    # Same personal sentences as Task 1
    sentences = [
        "Hello, how are you?",
        "My name is Anshul",
        "Purdue University is a great place to study Computer Science."
    ]

    # Initialize the model with 3 classes for Task A and 2 for Task B
    model = MultiTaskModel(num_classes_a=3, num_classes_b=2)

    # Set model to evaluation mode and disable gradient calculation
    model.eval()
    with torch.no_grad():
        logits_a, logits_b = model(sentences)

    # Print logits for Task A (sentence classification)
    print("\n=== Task A: Sentence Classification Logits ===\n")
    for i, logit in enumerate(logits_a):
        print(f"Sentence {i+1}: {sentences[i]}")
        print(f"Logits (Task A): {logit.tolist()}\n")

    # Print logits for Task B (sentiment analysis)
    print("\n=== Task B: Sentiment Analysis Logits ===\n")
    for i, logit in enumerate(logits_b):
        print(f"Sentence {i+1}: {sentences[i]}")
        print(f"Logits (Task B): {logit.tolist()}\n")

if __name__ == "__main__":
    test_multitask_model()
