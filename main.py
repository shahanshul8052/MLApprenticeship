# main.py

from models.sentence_transformer import SentenceTransformer

def run_sentence_encoding_demo():
    sentences = [
        "Hello, how are you?",
        "My name is Anshul",
        "Purdue University is a great place to study Computer Science."
    ]

    model = SentenceTransformer(pooling="mean")
    embeddings = model(sentences)

    print("\nSentence Embeddings:\n")
    for i, emb in enumerate(embeddings):
        print(f"Sentence {i+1}: {sentences[i]}")
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding (first 5 values): {emb[:5].tolist()}\n")

if __name__ == "__main__":
    run_sentence_encoding_demo()
