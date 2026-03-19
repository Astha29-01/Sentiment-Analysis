from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Simple dataset
texts = [
    "I love this movie",
    "This is amazing",
    "Very good experience",
    "I hate this",
    "This is bad",
    "Worst experience ever"
]

labels = [1,1,1,0,0,0]  # 1=positive, 0=negative

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Save model
pickle.dump((model, vectorizer), open("model.pkl", "wb"))

print("Model saved!")