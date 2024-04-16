import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargar datos desde el archivo CSV
df_music = pd.read_csv('music\\MusicDataset.csv')
df_music = df_music.dropna()
# Guardar las columnas originales para restaurarlas más tarde
artist_name_original = df_music['Artist Name']
track_name_original = df_music['Track Name']

# Normalización características categóricas
def label_encoder(datas, data_category):
    if datas[data_category].dtype == 'object':
        le = LabelEncoder()
        le.fit(datas[data_category].unique())
        datas[data_category] = le.transform(datas[data_category])

# Lista de variables categóricas
categorical_variables = ['Artist Name', 'Track Name']
for var in categorical_variables:
    label_encoder(df_music, var)

# Guardar el DataFrame con características normalizadas en un nuevo archivo CSV
df_music.to_csv('music\\music_normalized.csv', index=False)