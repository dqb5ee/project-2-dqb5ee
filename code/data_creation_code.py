import pandas as pd
import unicodedata
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading data
df_perf = pd.read_csv('/content/drive/MyDrive/DS_data/player_totals.csv', skiprows=6)
player_name_col_actual = df_perf.columns[2]
df_perf.rename(columns={player_name_col_actual: 'Player'}, inplace=True)

ast_col_actual = df_perf.columns[24]
pts_col_actual = df_perf.columns[29]
df_perf.rename(columns={ast_col_actual: 'AST', pts_col_actual: 'PTS'}, inplace=True)

df_sal = pd.read_csv('/content/drive/MyDrive/DS_data/NBA_Player_Salaries_2024-25_1.csv')

# Cleaning and merging
def normalize_name(name):
    if not isinstance(name, str): return name
    # Standardizing names
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii').strip().lower()

df_perf['Player_Key'] = df_perf['Player'].apply(normalize_name)
df_sal['Player_Key'] = df_sal['Player'].apply(normalize_name)

# Merging datasets on cleaned names
df = pd.merge(df_perf, df_sal, on='Player_Key', how='inner', suffixes=('', '_sal'))

# Converting salary string to float
df['Salary_Num'] = df['Salary'].replace(r'[$,,]', '', regex=True).astype(float)
df = df[['Player', 'PTS', 'AST', 'Salary_Num']].dropna()
