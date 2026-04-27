# DS 4320 Project 2: The Value Gap Pipeline
### Using K-Means Clustering to Expose Salary Inefficiencies in the 2024–25 NBA Season

### Setup


```python
!pip install pymongo
```

    Collecting pymongo
      Downloading pymongo-4.17.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (10 kB)
    Collecting dnspython<3.0.0,>=2.6.1 (from pymongo)
      Downloading dnspython-2.8.0-py3-none-any.whl.metadata (5.7 kB)
    Downloading pymongo-4.17.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (1.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m20.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dnspython-2.8.0-py3-none-any.whl (331 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m331.1/331.1 kB[0m [31m14.1 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: dnspython, pymongo
    Successfully installed dnspython-2.8.0 pymongo-4.17.0



```python
# Importing all required libraries
import logging
import unicodedata

import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from google.colab import userdata

import plotly.express as px
import plotly.graph_objects as go

# Configure logging to write to a log file for traceability
logging.basicConfig(
    filename='pipeline.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info('Pipeline started.')
print('Setup complete. Logging to pipeline.log.')
```

    Setup complete. Logging to pipeline.log.


### Data Preparation and MongoDB Query


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
MONGO_URI = userdata.get('MONGO_URI')
DB_NAME = "Cluster2"

try:
    # Connect to MongoDB Atlas
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    logging.info(f'Connected to MongoDB database: {DB_NAME}')

    # --- Load data from CSVs and insert into MongoDB --- #
    # Load player totals from CSV
    df_perf_csv = pd.read_csv('/content/drive/MyDrive/DS_data/player_totals.csv', skiprows=6)
    player_name_col_actual = df_perf_csv.columns[2]
    df_perf_csv.rename(columns={player_name_col_actual: 'Player'}, inplace=True)
    ast_col_actual = df_perf_csv.columns[24]
    pts_col_actual = df_perf_csv.columns[29]
    df_perf_csv.rename(columns={ast_col_actual: 'AST', pts_col_actual: 'PTS'}, inplace=True)

    # Load player salaries from CSV
    df_sal_csv = pd.read_csv('/content/drive/MyDrive/DS_data/NBA_Player_Salaries_2024-25_1.csv')

    # Clear existing collections before inserting to prevent duplicates on re-run
    db['player_totals'].delete_many({})
    db['player_salaries'].delete_many({})
    logging.info('Cleared existing player_totals and player_salaries collections.')

    # Insert data into MongoDB collections
    db['player_totals'].insert_many(df_perf_csv.to_dict('records'))
    db['player_salaries'].insert_many(df_sal_csv.to_dict('records'))
    logging.info(f'Inserted {len(df_perf_csv)} documents into player_totals.')
    logging.info(f'Inserted {len(df_sal_csv)} documents into player_salaries.')
    # --- End of CSV load and MongoDB insert --- #

    # Query player totals collection (now populated from CSVs) — pulls all documents into a list
    totals_cursor = db['player_totals'].find({}, {'_id': 0})
    df_perf = pd.DataFrame(list(totals_cursor))
    logging.info(f'Loaded df_perf from MongoDB: {len(df_perf)} documents')

    # Query player salaries collection (now populated from CSVs)
    salaries_cursor = db['player_salaries'].find({}, {'_id': 0})
    df_sal = pd.DataFrame(list(salaries_cursor))
    logging.info(f'Loaded df_sal from MongoDB: {len(df_sal)} documents')

    print(f'player_totals: {len(df_perf)} rows')
    print(f'player_salaries: {len(df_sal)} rows')

except Exception as e:
    logging.error(f'MongoDB connection or data loading failed: {e}')
    raise
```

    player_totals: 734 rows
    player_salaries: 563 rows



```python
def normalize_name(name):
    """
    Normalize a player name for consistent merging.
    Converts accented characters to ASCII equivalents,
    strips whitespace, and lowercases the result.
    Example: 'Alperen Şengün' -> 'alperen sengun'
    """
    if not isinstance(name, str):
        return name
    return (
        unicodedata.normalize('NFKD', name)
        .encode('ascii', 'ignore')
        .decode('ascii')
        .strip()
        .lower()
    )


try:
    # Check if DataFrames are empty due to no data from MongoDB
    if df_perf.empty or df_sal.empty:
        logging.error('One or both input DataFrames (df_perf, df_sal) are empty. Cannot proceed with data cleaning.')
        print('Error: No player data or salary data found. Check MongoDB collections.')
        raise ValueError('No data found in player_totals or player_salaries MongoDB collections.')

    # Drop duplicate players — keep only the season total row (first occurrence)
    # Basketball Reference lists traded players once per team plus a combined total
    df_perf = df_perf.drop_duplicates(subset='Player', keep='first').copy()

    # Create normalized join key on both DataFrames
    df_perf['Player_Key'] = df_perf['Player'].apply(normalize_name)
    df_sal['Player_Key'] = df_sal['Player'].apply(normalize_name)

    # Deduplicate salaries — keep first occurrence per player
    df_sal = df_sal.drop_duplicates(subset='Player_Key', keep='first').copy()

    # Inner join: only keep players present in both datasets
    # This drops players with no salary data and vice versa
    df = pd.merge(df_perf, df_sal[['Player_Key', 'Salary']], on='Player_Key', how='inner')

    # Clean salary: strip $ and commas, convert to float
    df['Salary_Num'] = (
        df['Salary']
        .astype(str)
        .str.replace(r'[$,]', '', regex=True)
        .astype(float)
    )

    # Keep only the columns needed for clustering and visualization
    df = df[['Player', 'PTS', 'AST', 'Salary_Num']].dropna()

    logging.info(f'Merged dataset ready: {len(df)} players')
    print(f'Final merged dataset: {len(df)} players')
    df.head()

except Exception as e:
    logging.error(f'Data cleaning failed: {e}')
    raise
```

    Final merged dataset: 487 players


### K-Means Clustering


```python
try:
    # Scale PTS and AST to equal variance before clustering
    # Without this, PTS (larger numbers) dominates the distance metric
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['PTS', 'AST']])

    # Fit k-means with k=3 and a fixed random seed for reproducibility
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(scaled_features)

    # Evaluate cluster quality — higher silhouette = better separation
    sil = silhouette_score(scaled_features, df['cluster_id'])
    logging.info(f'K-means fitted. Silhouette score: {sil:.3f}')
    print(f'Silhouette Score: {sil:.3f}  (target > 0.3 for meaningful clusters)')

    # Label clusters by average PTS so High = best scorers consistently
    label_order = df.groupby('cluster_id')['PTS'].mean().sort_values().index
    label_map = {label_order[0]: 'Low', label_order[1]: 'Medium', label_order[2]: 'High'}
    df['Cluster'] = df['cluster_id'].map(label_map)

    print('\nCluster distribution:')
    print(df['Cluster'].value_counts())

except Exception as e:
    logging.error(f'Clustering failed: {e}')
    raise
```

    Silhouette Score: 0.511  (target > 0.3 for meaningful clusters)
    
    Cluster distribution:
    Cluster
    Medium    217
    Low       198
    High       72
    Name: count, dtype: int64



```python
try:
    # Identify signing targets: High cluster players below median salary
    # These players produce at an elite level but cost below market rate
    high_perf = df[df['Cluster'] == 'High'].copy()
    avg_pts = high_perf['PTS'].mean()
    avg_ast = high_perf['AST'].mean()

    # Filter for above-average producers within the High cluster
    # then sort by salary ascending to surface the best value
    value_targets = (
        high_perf[
            (high_perf['PTS'] > avg_pts) &
            (high_perf['AST'] > avg_ast)
        ]
        .sort_values('Salary_Num')
        .head(6)
    )

    # Identify avoid list: Medium cluster players in top third of salaries
    salary_67th = df['Salary_Num'].quantile(0.67)
    avoid_list = (
        df[
            (df['Cluster'] == 'Medium') &
            (df['Salary_Num'] >= salary_67th)
        ]
        .sort_values('Salary_Num', ascending=False)
        .head(6)
    )

    logging.info(f'Targets identified: {len(value_targets)} | Avoid list: {len(avoid_list)}')

    print('=== TOP SIGNING TARGETS: High Output, Low Cost ===')
    print(value_targets[['Player', 'PTS', 'AST', 'Salary_Num']].to_string(index=False))
    print('\n=== AVOID LIST: Medium Output, High Cost ===')
    print(avoid_list[['Player', 'PTS', 'AST', 'Salary_Num']].to_string(index=False))

except Exception as e:
    logging.error(f'Recommendation generation failed: {e}')
    raise
```

    === TOP SIGNING TARGETS: High Output, Low Cost ===
               Player  PTS  AST  Salary_Num
          Jalen Duren  251  807   4536840.0
       Alperen Şengün  209  786   5424654.0
            Zach Edey  182  548   5756880.0
    Jonas Valančiūnas  181  627   9900000.0
       Onyeka Okongwu  193  660  14000000.0
            Josh Hart  201  737  18144000.0
    
    === AVOID LIST: Medium Output, High Cost ===
           Player  PTS  AST  Salary_Num
    Stephen Curry   95  310  55761216.0
     Kevin Durant  104  374  51179021.0
     Bradley Beal  136  177  50203930.0
      Paul George  101  219  49205800.0
     Jaylen Brown  154  368  49205800.0
     Jimmy Butler   52  296  48798677.0


**Analysis Rationale**

K-means was chosen because the goal is to discover natural performance tiers without imposing a predefined definition of what makes a player valuable - unsupervised clustering is the right tool when the structure of the answer is unknown. k=3 produces three clean tiers (High, Medium, Low) that a non-technical front office can act on directly. StandardScaler is applied so that PTS and AST contribute equally to clustering, since without it the larger point totals would dominate the distance calculations and assists would have almost no influence. Season totals are used over per-game averages because absolute output matters more than efficiency for roster-building — a player averaging 25 points over 20 games contributes less sustained value than one averaging 18 over 80.

## Visualization


```python
try:
    # Base scatter plot: all players colored by salary, shaped by cluster
    fig = px.scatter(
        df,
        x='PTS',
        y='AST',
        color='Salary_Num',
        symbol='Cluster',
        hover_data=['Player'],
        title='The Value Gap: NBA Player Production vs. Salary (2024–25)',
        labels={
            'Salary_Num': 'Salary ($)',
            'PTS': 'Total Points',
            'AST': 'Total Assists',
            'Cluster': 'Performance Tier'
        },
        color_continuous_scale='Viridis'
    )

    # Overlay red stars for the top value targets
    fig.add_trace(
        go.Scatter(
            x=value_targets['PTS'],
            y=value_targets['AST'],
            mode='markers+text',
            marker=dict(
                symbol='star',
                size=16,
                color='red',
                line=dict(width=1, color='black')
            ),
            text=value_targets['Player'],
            textposition='top center',
            textfont=dict(size=10, family='Arial'),
            name='Top Targets',
            hoverinfo='skip'
        )
    )

    # Publication-quality layout settings
    fig.update_layout(
        title={
            'text': 'The Value Gap: NBA Player Production vs. Salary (2024–25)',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20, family='Arial')
        },
        font=dict(family='Arial', size=12),
        plot_bgcolor='rgba(245, 245, 245, 1)',
        paper_bgcolor='white',
        xaxis=dict(title='Total Points', gridcolor='white', showgrid=True),
        yaxis=dict(title='Total Assists', gridcolor='white', showgrid=True),
        coloraxis_colorbar=dict(
            title='Salary ($)',
            tickformat='~s',
            thickness=15,
            len=0.7,
            x=1.02
        ),
        legend=dict(title='Tier / Highlight', x=1.15, y=0.8),
        margin=dict(l=50, r=200, t=90, b=60),
        annotations=[
            dict(
                text='Source: Basketball Reference 2024–25 | ★ = Top Value Targets',
                xref='paper', yref='paper',
                x=0, y=-0.12,
                showarrow=False,
                font=dict(size=10, color='gray', family='Arial')
            )
        ]
    )

    fig.show()

except Exception as e:
    logging.error(f'Visualization failed: {e}')
    raise
```



                        })                };                            </script>        </div>
</body>
</html>



```python
from IPython.display import Image, display

image_path = "/content/drive/MyDrive/DS_data/newplot-3.png"
display(Image(filename=image_path))
```


    
![png](pipeline_files/pipeline_14_0.png)
    


**Visualization Rationale**

A scatter plot was chosen because it encodes all four variables at once: total points (x), total assists (y), cluster tier (shape), and salary (color). Viridis was selected because it is perceptually uniform and colorblind-friendly. Red stars are overlaid on the top targets so the key finding is immediately visible without requiring the reader to interpret the color scale.

### Model Validation and Log Completion


```python
# Model Validation and Sanity Check ---
# Verify that the clustering output makes domain sense before accepting results

try:
    print("=== Cluster Summary ===")
    summary = df.groupby('Cluster')[['PTS', 'AST', 'Salary_Num']].mean().round(0)
    print(summary)

    # Sanity check 1: High cluster should have highest avg PTS
    cluster_pts = df.groupby('Cluster')['PTS'].mean()
    assert cluster_pts['High'] > cluster_pts['Medium'] > cluster_pts['Low'], \
        "Cluster ordering by PTS is incorrect — check label mapping"
    print("\n✓ Cluster ordering validated: High > Medium > Low by average PTS")

    # Sanity check 2: Targets should all be in the High cluster
    assert all(value_targets['Cluster'] == 'High'), \
        "One or more targets are not in the High cluster"
    print("✓ All signing targets confirmed in High cluster")

    # Sanity check 3: Avoid list should all be in Medium cluster
    assert all(avoid_list['Cluster'] == 'Medium'), \
        "One or more avoid list players are not in the Medium cluster"
    print("✓ All avoid list players confirmed in Medium cluster")

    # Sanity check 4: Silhouette score acceptable
    assert sil > 0.3, f"Silhouette score {sil:.3f} is below acceptable threshold"
    print(f"✓ Silhouette score {sil:.3f} exceeds minimum threshold of 0.3")

    logging.info('Model validation passed all sanity checks.')

except AssertionError as e:
    logging.error(f'Model validation failed: {e}')
    raise

except Exception as e:
    logging.error(f'Unexpected error during validation: {e}')
    raise
```

    === Cluster Summary ===
               PTS    AST  Salary_Num
    Cluster                          
    High     170.0  531.0  19146406.0
    Low       34.0   71.0   4733844.0
    Medium   117.0  246.0  12720175.0
    
    ✓ Cluster ordering validated: High > Medium > Low by average PTS
    ✓ All signing targets confirmed in High cluster
    ✓ All avoid list players confirmed in Medium cluster
    ✓ Silhouette score 0.511 exceeds minimum threshold of 0.3



```python
# Log pipeline completion
logging.info('Pipeline completed successfully.')
print('Pipeline complete. Log written to pipeline.log.')
```

    Pipeline complete. Log written to pipeline.log.



```python
!jupyter nbconvert --to markdown "/content/drive/MyDrive/pipeline.ipynb" --output-dir="/content/"
```
