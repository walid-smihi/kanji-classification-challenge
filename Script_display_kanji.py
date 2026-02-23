import matplotlib.pyplot as plt


def display_kanji(kanji_vector):
    """
    Affiche la représentation graphique d'un caractère kanji à partir de sa forme vectorielle.

    Paramètres :
    - kanji_vector : Un tableau numpy de forme (4096,) représentant le caractère kanji aplati.
    """
    if kanji_vector.shape[0] != 4096:
        raise ValueError("Le vecteur d'entrée doit avoir 4096 éléments.")

    # Remodeler le vecteur en une matrice 64x64
    kanji_matrix = kanji_vector.reshape(64, 64)

    # Afficher le caractère kanji comme une image
    plt.imshow(kanji_matrix, cmap='gray')
    plt.axis('off')  # Désactiver les étiquettes et les graduations des axes
    plt.show()

# Exemple d'utilisation :
# Pour tester, on peut décommenter Script_visualisation_TSNE.py : ligne 23
# Sinon on peut aussi charger les données en début de script
# puis décommenter les lignes suivantes :
# kanji_vector = data[0]  # utilisation du premier kanji dans l'ensemble de données
# display_kanji(kanji_vector)
