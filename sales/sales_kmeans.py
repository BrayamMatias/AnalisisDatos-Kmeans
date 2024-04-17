import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('sales\\sales_correlacion.csv')
data = data.dropna()

# Determinación del número óptimo de clusters usando el método del codo
inertia_sales = []
for k in range(1, 100):
    kmeans_sales = KMeans(n_clusters=k, random_state=42)
    kmeans_sales.fit(data)
    inertia_sales.append(kmeans_sales.inertia_)

# "import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 8))
# plt.plot(range(1, 100), inertia_sales, 'bx-')
# plt.xlabel('Número de clusters')
# plt.ylabel('Inercia')
# plt.title('Método del codo para determinar el número óptimo de clusters')
# plt.show()"

# Basado en la gráfica, seleccionar el número óptimo de clusters y aplicar KMeans
k_optimo_sales = 25
kmeans_sales = KMeans(n_clusters=k_optimo_sales, random_state=42)
y_pred_sales = kmeans_sales.fit_predict(data)

# Agregar las etiquetas de los clústeres al DataFrame original
df_sales_filtered = data.dropna()  # Eliminar filas con valores faltantes en df_sales
df_sales_filtered['Cluster'] = y_pred_sales



# Ordenar los datos por cluster
df_sales_filtered = df_sales_filtered.sort_values(by='Cluster')

# Visualizar las canciones agrupadas por clúster y contar cuántos elementos hay en cada uno
for cluster in range(k_optimo_sales):
    cluster_sales = df_sales_filtered[df_sales_filtered['Cluster'] == cluster]
    cluster_size = len(cluster_sales)  # Contar el número de elementos en el cluster
    print(f"Cluster {cluster}: {cluster_size} elementos")


# Guardar los datos de los clusters en un nuevo archivo CSV
df_sales_filtered.to_csv('sales\\sales_clusters.csv', index=False)

