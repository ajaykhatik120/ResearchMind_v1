import numpy as np
from core.embeddings import get_embeddings

model = get_embeddings()

texts = [
    "A dog is playing.",
    "A puppy is running.",
    "Stock prices are rising."
]

vectors = model.embed_documents(texts)

print("Vector count:", len(vectors))
print("Dimensions:", len(vectors[0]))

similarity_1 = np.dot(vectors[0], vectors[1])
similarity_2 = np.dot(vectors[0], vectors[2])

print("Dog vs Puppy:", similarity_1)
print("Dog vs Stock:", similarity_2)