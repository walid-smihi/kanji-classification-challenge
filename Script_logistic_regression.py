from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charge les données d'entraînement
data_train = np.loadtxt("starting_k/kanji_train_data.csv", delimiter=",")
target_train = np.loadtxt("starting_k/kanji_train_target.csv")

# Formatage sur les données (Ajustement & transformation, moyenne, écart type)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_train)

# Divise les données formatées en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(data_scaled, target_train, test_size=0.2, random_state=42)

# Plage de valeurs pour C à tester
C_values = np.logspace(np.log10(0.0001), np.log10(0.5), 10)

accuracy_scores = []
# Boucle sur les valeurs de C
for C in C_values:
    log_reg = LogisticRegression(C=C, max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)
    print(f"C={C:.4f}, Validation Accuracy: {accuracy:.2f}")

f1_macro_scores = []
for C in C_values:
    log_reg = LogisticRegression(C=C, max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_macro_scores.append(f1_macro)
    print(f"C={C:.4f}, F1 Macro Score: {f1_macro:.2f}")

# Tracer la précision en fonction de C
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, accuracy_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Valeur de C (Inverse de la force de régularisation)')
plt.ylabel('Précision sur l\'ensemble de validation')
plt.title('Précision de la Régression Logistique en fonction de la valeur de C')
plt.grid(True)
plt.show()

# Tracer le score F1 Macro en fonction de C
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, f1_macro_scores, marker='o', linestyle='-', color='r')
plt.xlabel('Valeur de C (Inverse de la force de régularisation)')
plt.ylabel('Score F1 Macro')
plt.title('Score F1 Macro de la Régression Logistique en fonction de la valeur de C')
plt.grid(True)
plt.show()

# Après avoir choisi la meilleure valeur de C, entraînez le modèle final
optimal_C = 0.0007 # à adapter selon le résultat du graphe
log_reg_final = LogisticRegression(C=optimal_C, max_iter=1000, random_state=42)
log_reg_final.fit(X_train, y_train)



# Charge les données de test
data_test = np.loadtxt("kanji_test_data.csv", delimiter=",")

# Formate les données de test de la même manière que pour les données d'entraînement
data_test_scaled = scaler.transform(data_test)

# Utilise le modèle pour faire des prédictions sur les données de test
predictions = log_reg_final.predict(data_test_scaled)

# Enregistre les prédictions dans un fichier CSV
np.savetxt("kanji_test_predictions.csv", predictions, fmt='%d', delimiter=",")
print("Predictions saved to 'kanji_test_predictions.csv'.")
