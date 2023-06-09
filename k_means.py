import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt

# Wczytanie zbioru danych Wine
wine = datasets.load_wine()
X = wine.data

# Opis zbioru danych
print(wine.DESCR)

# Lista do przechowywania wyników silhouette
silhouette_scores = []

# Wykonanie klasteryzacji K-means dla k=2, k=3 i k=4
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette = silhouette_score(X, labels)
    silhouette_scores.append(silhouette)

    # Generowanie raportu klasteryzacji
    cluster_report = pd.Series(labels).value_counts()
    print(f"\nRaport klasteryzacji dla k={k}:")
    print(cluster_report)

    # Zapisywanie raportu klasteryzacji do pliku
    with open(f"cluster_report_k{k}.txt", "w") as file:
        file.write(str(cluster_report))

    # Generowanie wykresu klasteryzacji
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.title(f"Wykres klasteryzacji dla k={k}")
    plt.show()

    # Generowanie wykresu silhouette
    visualizer = SilhouetteVisualizer(kmeans)
    visualizer.fit(X)
    visualizer.show()

# Wykres silhouette dla k=2, k=3 i k=4
plt.figure(figsize=(10, 5))
plt.plot([2, 3, 4], silhouette_scores, marker='o')
plt.title("Wykres silhouette dla k=2, k=3 i k=4")
plt.xlabel("Liczba klastrów")
plt.ylabel("Wartość silhouette")
plt.show()

# Zapisywanie wyników silhouette do pliku
with open("silhouette_scores.txt", "w") as file:
    for k, score in zip([2, 3, 4], silhouette_scores):
        file.write(f"Silhouette score dla k={k}: {score}\n")
