import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargar datos desde el archivo CSV
df_sales = pd.read_csv('sales\\SalesDataset.csv')
df_sales = df_sales.dropna()


# Normalización características categóricas
def label_encoder(datas, data_category):
    if datas[data_category].dtype == 'object':
        le = LabelEncoder()
        le.fit(datas[data_category].unique())
        datas[data_category] = le.transform(datas[data_category])

# Lista de variables categóricas
categorical_variables = ['Date', 'Year', 'Month', 'Customer Gender', 'Country', 'State', 'Product Category', 'Sub Category']
for var in categorical_variables:
    label_encoder(df_sales, var)

# Guardar el DataFrame con características normalizadas en un nuevo archivo CSV
df_sales.to_csv('sales\\sales_normalized.csv', index=False)