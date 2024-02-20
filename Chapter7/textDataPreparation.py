from sklearn.feature_extraction.text import CountVectorizer


text = ["The quick brown fox jumped over the lazy dog."]
vectorizer = CountVectorizer()
vectorizer.fit(text) # create a vocabulary from text input

print("vocab=",vectorizer.vocabulary_)

vector = vectorizer.transform(text)

print(vector.shape)
print(vector.toarray())

vector = vectorizer.transform(["the brown fox and big puppy"])
print(vector.shape)
print(vector.toarray())

vector = vectorizer.transform(["the brown fox","lazy dog jumped"])
print(vector.shape)
print(vector.toarray())

