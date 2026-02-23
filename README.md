# Walid Smihi Challenge

Projet Python de classification de kanjis (KNN, régression logistique, réseau de neurones MLP et CNN) avec visualisation t-SNE.

## Contenu

- `Find_the_optimal_k.py` : recherche d'un `k` pertinent pour KNN
- `Script_knn_classification_test.py` : prédiction KNN sur le jeu de test
- `Script_logistic_regression.py` : entraînement + validation d'une régression logistique
- `neural_network_classifier.py` : entraînement + validation d'un MLP
- `convolutional_neural_network.py` : entraînement + validation d'un CNN (PyTorch)
- `Script_visualisation_TSNE.py` : projection t-SNE des données
- `Script_display_kanji.py` : affichage d'un kanji 64x64
- `Rapport.pdf` : rapport du projet

## Prérequis

- Python 3.10+

Installation des dépendances :

```bash
pip install -r requirements.txt
```

## Données attendues

Le code attend les fichiers suivants :

- `starting_k/kanji_train_data.csv`
- `starting_k/kanji_train_target.csv`
- `kanji_test_data.csv` (à la racine du projet)

Les scripts de prédiction génèrent :

- `kanji_test_predictions.csv`

## Exécution

Exemple :

```bash
python Script_knn_classification_test.py
```

Tu peux lancer de la même manière les autres scripts `.py`.

## Publication sur GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <URL_DU_REPO_GITHUB>
git push -u origin main
```

## Remarque

Ce dépôt contient des scripts Python (un projet de code), pas un exécutable binaire.
