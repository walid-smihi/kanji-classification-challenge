from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Étape 1: Charger les données d'entraînement et les cibles correspondantes
data_train = np.loadtxt("starting_k/kanji_train_data.csv", delimiter=",")
target_train = np.loadtxt("starting_k/kanji_train_target.csv")

# Étape 2: Charger les données de test (sans les cibles correspondantes, car c'est un ensemble de test)
data_test = np.loadtxt("kanji_test_data.csv", delimiter=",")

# Étape 3: Créer et entraîner le modèle KNN avec k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data_train, target_train)

# Étape 4: Utiliser le modèle pour faire des prédictions sur les données de test
predictions = knn.predict(data_test)

np.savetxt("kanji_test_predictions.csv", predictions, delimiter=",")