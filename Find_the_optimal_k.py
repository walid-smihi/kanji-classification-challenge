from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Chargement des données
data = np.loadtxt("starting_k/kanji_train_data.csv", delimiter=",")
target = np.loadtxt("starting_k/kanji_train_target.csv")

# Step 2: Prétraitement des données
X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=42)

# espaces réservés pour stocker la précision pour les ensembles de validation et de formation
train_accuracies = []
val_accuracies = []
k_values = range(1, 16)  # Testing values of k from 1 to 15

# Step 3: Affichage des différents k et des précisions pour chaque
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # données d'entrainement
    train_predictions = knn.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_accuracies.append(train_accuracy)

    # données de validation
    val_predictions = knn.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_accuracies.append(val_accuracy)

    # affichage de la précision pour les deux
    print(f"k = {k}: Training Accuracy = {train_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}")

# Step 4: affichage du graphe pour selectionner le k le plus optimale (a vue d'oeil)
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(k_values, val_accuracies, marker='s', label='Validation Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Training vs. Validation Data')
plt.legend()
plt.show()
