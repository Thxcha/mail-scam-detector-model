import numpy as np
import math

class TfidfVectorizer:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = None

    def fit(self, documents):
        n_documents = len(documents)
        df = {}
        for doc in documents:
            tokens = set(doc.split())
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        self.vocabulary_ = {token: idx for idx, token in enumerate(df.keys())}
        self.idf_ = np.zeros(len(self.vocabulary_))
        for token, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((n_documents + 1) / (df[token] + 1)) + 1
        return self

    def transform(self, documents):
        n_documents = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_documents, n_features))
        for i, doc in enumerate(documents):
            tokens = doc.split()
            tf = {}
            for token in tokens:
                if token in self.vocabulary_:
                    tf[token] = tf.get(token, 0) + 1
            for token, count in tf.items():
                idx = self.vocabulary_[token]
                X[i, idx] = count * self.idf_[idx]
        return X

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            X_c = X[np.array(y) == cls]
            self.class_log_prior_[cls] = np.log(X_c.shape[0] / X.shape[0])
            count = X_c.sum(axis=0) + self.alpha
            total_count = count.sum()
            self.feature_log_prob_[cls] = np.log(count / total_count)
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for cls in self.classes_:
                score = self.class_log_prior_[cls] + np.sum(x * self.feature_log_prob_[cls])
                class_scores[cls] = score
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)
        return np.array(predictions)


class Pipeline:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, X, y):
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_tfidf, y)
        return self

    def predict(self, X):
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict(X_tfidf)

