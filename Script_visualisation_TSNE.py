import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import Script_display_kanji

data = np.loadtxt("starting_k/kanji_train_data.csv", delimiter=",")
target = np.loadtxt("starting_k/kanji_train_target.csv")

nb_kanji_plot = 1000

data_embedded = TSNE(n_components=2).fit_transform(data[:nb_kanji_plot])

print(data_embedded.shape)

print(target.shape)

for i in range(20):

    plt.scatter(data_embedded[target[:nb_kanji_plot] == i,0],data_embedded[target[:nb_kanji_plot] == i,1])

plt.show()

# Script_display_kanji.display_kanji(data[0])



