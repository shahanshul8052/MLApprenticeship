# ML Apprentice Take-Home Assessment
# logit -- raw unnormalized output of a model before applying a probability function like sigmoid, and then pass through function to get actual probabilities 
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

## Task 2: Multi-Task Learning Expansion

For Task 2, I extended the sentence transformer model to support multi-task learning. The idea behind multi-task learning is to train a single shared model on multiple related tasks, which can improve generalization and reduce the number of parameters compared to training separate models for each task.

### Model Architecture

I built a class called `MultiTaskModel` using PyTorch. The core of the model is the same transformer encoder from Task 1 (`distilbert-base-uncased`), which acts as a shared feature extractor. I then added two task-specific heads on top of the shared encoder for our 2 differnen tasks:

- **Task A – Sentence Classification**: This head is a linear layer that outputs logits for 3 made-up classes. It could represent something like topic classification (e.g., "greeting", "identity", "education").
- **Task B – Sentiment Analysis**: This head outputs logits for 2 classes (positive or negative sentiment). I used a simple linear classifier here as well.

### Pooling Strategy

Just like in Task 1, I used **mean pooling** to convert token-level embeddings into a fixed-size sentence embedding. This pooled embedding is passed into both task heads. I stuck with mean pooling because it’s robust and performs well for sentence-level tasks without requiring extra layers or complexity.

### Inference & Testing

I wrote a separate script, `test_multitask.py`, to test the model with the same sentences from Task 1. The script prints the logits produced by each task head. Here’s an example of what the output looked like:

=== Task A: Sentence Classification Logits ===

Sentence 1: Hello, how are you?
Logits (Task A): [0.16352327167987823, 0.2980576455593109, -0.3741283416748047]

Sentence 2: My name is Anshul
Logits (Task A): [0.0938272476196289, 0.14478030800819397, -0.24297648668289185]

Sentence 3: Purdue University is a great place to study Computer Science.
Logits (Task A): [0.2300703078508377, 0.20275598764419556, -0.34060847759246826]


=== Task B: Sentiment Analysis Logits ===

Sentence 1: Hello, how are you?
Logits (Task B): [-0.4396241009235382, -0.07834406197071075]

Sentence 2: My name is Anshul
Logits (Task B): [-0.26447242498397827, -0.06079651415348053]

Sentence 3: Purdue University is a great place to study Computer Science.
Logits (Task B): [-0.23500744998455048, 0.114423468708992]

# Task 3 Writeup -- Training Considerations

The task asked me to think througha some different training scenarios and how to apply transfer learning if we had to. 

Freezing Scenarios -
#### 1. Freeze the entire network

If we freeze both the transformer and task heads, the model is only useful for interference and becomes static. It basically turns into a fixed feature extractor. This means I would still be able to pass new sentences and get predictions but wouldn't be able to improve the model by training it. This could be useful if I want to reuse a pre trained model on new data without any change, but would not helpif the new tasks are different compared to the original trained task 

#### 2. If only the transformer backbone should be frozen

When I was watching tutorials to learn, this was very common. The idea behind it was that the transformer encoder like DistilBERT had already learned a lot of general language understanding from a huge dataset. So instead of retraining the whole transformer, I can keep the weights frozen and just train the task specific layers that I added for A and B. This saves training time and works well without a lot of labeled data 

#### 3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.

If I freeze a task specific head like Task A or B, this means that I trust that part of the model and don't want any change to occur to it. If I know the task is performing well, then tis would be a great idea and I would want to improve the other task, as this would help avoid overfitting while learning on the other. 

### Transfer Learning

This is how I would approach each area if I were fine-tuning this model: 
- Pre-trained model: Since the `distilbert-base-uncased` has previously been trained on a substantial corpus of texts and is effective, I would utilize it.
The transformer's top layers should be unfrozen, while the lower layers should be frozen. If this was the case, it allows the model to adjust to my particular needs while retaining some of the broad information it already possesses (such as sentence structure or word correlations that it has already learned).
- Why don't we freeze everything? 
Excessive freezing prevents the model from learning anything new. However, overfitting can also result from not freezing anything, particularly if my dataset is too little or deviates from what the model predicts.

