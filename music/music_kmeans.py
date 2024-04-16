import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('music\\music_correlacion.csv')
data = data.dropna()

# Determinación del número óptimo de clusters usando el método del codo
inertia_music = []
for k in range(1, 11):
    kmeans_music = KMeans(n_clusters=k, random_state=42)
    kmeans_music.fit(data)
    inertia_music.append(kmeans_music.inertia_)

# Basado en la gráfica, seleccionar el número óptimo de clusters y aplicar KMeans
k_optimo_music = 10
kmeans_music = KMeans(n_clusters=k_optimo_music, random_state=42)
y_pred_music = kmeans_music.fit_predict(data)

# Agregar las etiquetas de los clústeres al DataFrame original
df_music_filtered = data.dropna()  # Eliminar filas con valores faltantes en df_music
df_music_filtered['Cluster'] = y_pred_music



# Ordenar los datos por cluster
df_music_filtered = df_music_filtered.sort_values(by='Cluster')

# Visualizar las canciones agrupadas por clúster y contar cuántos elementos hay en cada uno
for cluster in range(k_optimo_music):
    cluster_music = df_music_filtered[df_music_filtered['Cluster'] == cluster]
    cluster_size = len(cluster_music)  # Contar el número de elementos en el cluster
    print(f"Cluster {cluster}: {cluster_size} elementos")
    print(cluster_music[['Class']])
    print()

# Guardar los datos de los clusters en un nuevo archivo CSV
df_music_filtered.to_csv('music\\music_clusters.csv', index=False)
