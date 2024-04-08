
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import numpy as np

class ProtoNetCLMBRClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        # (n_patients, clmbr_embedding_size)
        n_classes = len(set(y_train))
        
        # (n_classes, clmbr_embedding_size)
        self.prototypes = np.zeros((n_classes, X_train.shape[1]))
        for cls in range(n_classes):
            indices = np.nonzero(y_train == cls)[0]
            examples = X_train[indices]
            self.prototypes[cls, :] = np.mean(examples, axis=0)

    def predict_proba(self, X_test):
        # (n_patients, clmbr_embedding_size)    
        dists = pairwise_distances(X_test, self.prototypes, metric='euclidean')
        # Negate distance values
        neg_dists = -dists

        # Apply softmax function to convert distances to probabilities
        probabilities = np.exp(neg_dists) / np.sum(np.exp(neg_dists), axis=1, keepdims=True)

        return probabilities

    def predict(self, X_train):
        dists = self.predict_proba(X_train)
        predictions = np.argmax(dists, axis=1)
        return predictions
    
    def save_model(self, model_save_dir, model_name):
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok = True)
        np.save(self.prototypes, os.path.join(model_save_dir, f'{model_name}.npy'))

