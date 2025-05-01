# ML Apprentice Take-Home Assessment

# Goal

The goal of this assessment is to work with neural network architectures, specifically transformers and multi-task learning

# Task 1: Sentence Transformer Implementation

For the first task, I created a sentence transformer using PyTorch and the Hugging Face Transformers library. The purpose of this model is to encode sentences into embeddings that can be used for other downstream NLP tasks.

# Model Design and Key Choices

I used the `distilbert-base-uncased` transformer as the base model. This is a lighter version of BERT that offers good performance while being faster to run. I chose it to keep the model efficient and simple.

The sentence transformer wraps the Hugging Face model and uses a pooling strategy to create fixed-length sentence embeddings. I used mean pooling as a test, which averages the token embeddings across the sentence. This gives a more reliable sentence-level representation than just using the `[CLS]` token, especially when comparing sentences or using embeddings for classification.

## Testing the Model

I wrote a test script in `main.py` that runs the model on three sample sentences. The script prints out the shape of each embedding and a preview of its values. This helps verify that the model is working correctly.

Here are some sample outputs from running the test:

Sentence Embeddings:

Sentence 1: Hello, how are you?
Embedding shape: torch.Size([768])
Embedding (first 5 values): [-0.14021293818950653, -0.28772905468940735, 0.10716602951288223, -0.22320407629013062, 0.07658670842647552]

Sentence 2: My name is Anshul
Embedding shape: torch.Size([768])
Embedding (first 5 values): [0.08390495181083679, 0.05580494925379753, -0.21389839053153992, -0.5246104598045349, 0.029893305152654648]

Sentence 3: Purdue University is a great place to study Computer Science.
Embedding shape: torch.Size([768])
Embedding (first 5 values): [0.06927134841680527, 0.1761387437582016, 0.01445331983268261, 0.3833419382572174, 0.4592837691307068]