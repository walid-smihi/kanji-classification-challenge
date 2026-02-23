import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Vérifier si CUDA est disponible et sélectionner le device
print(torch.cuda.get_device_name(0))  # 0 corresponds to the first GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger et préparer les données
data_train = np.loadtxt("starting_k/kanji_train_data.csv", delimiter=",", dtype=np.float32)
target_train = np.loadtxt("starting_k/kanji_train_target.csv", delimiter=",", dtype=np.longlong)

# Normalisation des données
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)

# Reshape des données pour CNN ([num_samples, channels, height, width])
data_train_reshaped = data_train_scaled.reshape(-1, 1, 64, 64)  # Exemple de données 64x64

# Conversion en tenseurs PyTorch
X_train = torch.tensor(data_train_reshaped)
y_train = torch.tensor(target_train)

# Création d'un ensemble de données et séparation en entraînement et validation
dataset = TensorDataset(X_train, y_train)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Définition du modèle CNN
class KanjiCNN(nn.Module):
    def __init__(self):
        super(KanjiCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 1000)
        self.fc2 = nn.Linear(1000, 20)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialisation du modèle, du critère de perte et de l'optimiseur
model = KanjiCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
num_epochs = 10
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total_val += y_batch.size(0)
            correct_val += (predicted == y_batch).sum().item()
    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(
        f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Affichage de la précision d'entraînement vs précision de validation
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Charge les données de test
data_test = np.loadtxt("kanji_test_data.csv", delimiter=",", dtype=np.float32)

# formate les données de test
data_test_scaled = scaler.transform(data_test)

# ajuste les données de test pour CNN
data_test_reshaped = data_test_scaled.reshape(-1, 1, 64, 64)

# Convertit en PyTorch tensors
X_test = torch.tensor(data_test_reshaped).to(device)


model.eval()

# prédictions
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

# conversion des predictions en numpy array
predicted_numpy = predicted.cpu().numpy()

# enregistre les prédictions dans le fichier .csv
np.savetxt("kanji_test_predictions.csv", predicted_numpy, delimiter=",", fmt='%d')
print("Predictions saved to 'kanji_test_predictions.csv'.")