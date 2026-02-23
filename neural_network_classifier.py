from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charge les données d'entraînement
data_train = np.loadtxt("starting_k/kanji_train_data.csv", delimiter=",")
target_train = np.loadtxt("starting_k/kanji_train_target.csv", delimiter=",")

# Normalise les données d'entraînement
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)

# Divise les données en ensembles d'entraînement et de validation
X_train, X_test, y_train, y_test = train_test_split(data_train_scaled, target_train, test_size=0.2, random_state=42)

# Configurations de couches cachées à tester
hidden_layer_sizes = [(50,), (100,), (100, 50), (150, 100, 50), (200, 150, 100, 50)]

# Initialise les listes pour enregistrer les résultats
accuracies = []
f1_scores = []

# Itere sur les configurations de couches cachées
for size in hidden_layer_sizes:
    mlp = MLPClassifier(hidden_layer_sizes=size, max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    test_predictions = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions, average='macro')
    accuracies.append(accuracy)
    f1_scores.append(f1)
    print(f"Configuration {size}, Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.2f}")

# Crée des labels pour les configurations de couches cachées
# Conversion des configurations en chaînes pour l'axe X
config_labels = [f"{size}" for size in hidden_layer_sizes]

plt.figure(figsize=(14, 6))

# Précision par configuration
plt.subplot(1, 2, 1)
plt.plot(config_labels, accuracies, 'o-', color='blue')
plt.ylim(0.8, 1)  # Définir la limite de l'axe Y entre 0.8 et 1
plt.title('Accuracy per Configuration')
plt.xlabel('Configuration')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# Score F1 par configuration
plt.subplot(1, 2, 2)
plt.plot(config_labels, f1_scores, 'o-', color='orange')
plt.ylim(0.8, 1)  # Définir la limite de l'axe Y entre 0.8 et 1
plt.title('F1 Score per Configuration')
plt.xlabel('Configuration')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Utilise la meilleure configuration pour les prédictions réelles
best_config_index = np.argmax(f1_scores)
best_config = hidden_layer_sizes[best_config_index]
print(f"Using best configuration: {best_config}")

mlp_best = MLPClassifier(hidden_layer_sizes=best_config, max_iter=500, random_state=42)
mlp_best.fit(X_train, y_train)

# Charge et prépare les données de test
data_test = np.loadtxt("kanji_test_data.csv", delimiter=",")
data_test_scaled = scaler.transform(data_test)

# Faire des prédictions avec la meilleure configuration
real_test_predictions = mlp_best.predict(data_test_scaled)

# Sauvegarde les prédictions dans un fichier CSV
np.savetxt("kanji_test_predictions.csv", real_test_predictions, fmt='%d', delimiter=",")
print("Predictions saved to 'kanji_test_predictions.csv'.")
