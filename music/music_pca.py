import pandas as pd
from sklearn.decomposition import PCA

# Cargar el conjunto de datos desde un archivo CSV
data = pd.read_csv('music\\music_normalized.csv')

# Separar las características (X) de las etiquetas (y)
X = data.drop(columns=['Artist Name', 'Track Name', 'Class'])

# Instanciar el objeto PCA con el número deseado de componentes
pca = PCA(n_components=2)

# Ajustar y transformar los datos
X_pca = pca.fit_transform(X)

# Crear un nuevo DataFrame con las componentes principales
pca_df = pd.DataFrame(data=X_pca, columns=['Componente Principal 1', 'Componente Principal 2'])

# Concatenar las etiquetas (si las hay) al DataFrame reducido

# Guardar el DataFrame reducido en un nuevo archivo CSV
pca_df.to_csv('music\\music_pca.csv', index=False)