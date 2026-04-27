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


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="4de15b87-5964-4202-bc2a-2d24f0abd32c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("4de15b87-5964-4202-bc2a-2d24f0abd32c")) {                    Plotly.newPlot(                        "4de15b87-5964-4202-bc2a-2d24f0abd32c",                        [{"customdata":[["Anthony Edwards"],["Nikola Jokić"],["Giannis Antetokounmpo"],["Jayson Tatum"],["Devin Booker"],["Cade Cunningham"],["James Harden"],["Karl-Anthony Towns"],["LeBron James"],["Jaren Jackson Jr."],["Pascal Siakam"],["Jalen Williams"],["Alperen Şengün"],["Bam Adebayo"],["Michael Porter Jr."],["Nikola Vučević"],["Ivica Zubac"],["Domantas Sabonis"],["OG Anunoby"],["Desmond Bane"],["Evan Mobley"],["Miles Bridges"],["Julius Randle"],["Anthony Davis"],["Scottie Barnes"],["Deni Avdija"],["Christian Braun"],["Bennedict Mathurin"],["Naz Reid"],["Myles Turner"],["Victor Wembanyama"],["Jarrett Allen"],["Dyson Daniels"],["Dillon Brooks"],["Josh Hart"],["Brook Lopez"],["Josh Giddey"],["Tobias Harris"],["Jaden McDaniels"],["Russell Westbrook"],["Onyeka Okongwu"],["Amen Thompson"],["Keegan Murray"],["Jalen Duren"],["Kelly Oubre Jr."],["Toumani Camara"],["Alex Sarr"],["Rudy Gobert"],["Jonas Valančiūnas"],["Keon Johnson"],["P.J. Washington"],["Jakob Poeltl"],["Bub Carrington"],["Guerschon Yabusele"],["Nic Claxton"],["Luguentz Dort"],["Daniel Gafford"],["Kyle Filipowski"],["Yves Missi"],["Walker Kessler"],["Isaiah Hartenstein"],["Wendell Carter Jr."],["Draymond Green"],["Zach Edey"],["Kel'el Ware"],["Nick Richards"],["Goga Bitadze"],["Clint Capela"],["Donovan Clingan"],["Isaiah Stewart"],["Kevon Looney"],["Mason Plumlee"]],"hovertemplate":"Performance Tier=High\u003cbr\u003eTotal Points=%{x}\u003cbr\u003eTotal Assists=%{y}\u003cbr\u003ePlayer=%{customdata[0]}\u003cbr\u003eSalary ($)=%{marker.color}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"High","marker":{"color":[42176400.0,51415938.0,48787676.0,34848340.0,49205800.0,13940809.0,33653846.0,49205800.0,48728845.0,25257798.0,42176400.0,4775760.0,5424654.0,34848340.0,35859950.0,20000000.0,11743210.0,40500000.0,36637932.0,34005250.0,11227657.0,27173913.0,33073920.0,43219440.0,10130980.0,15625000.0,3089640.0,7245720.0,13986432.0,19928500.0,12768960.0,20000000.0,6059520.0,22255493.0,18144000.0,23000000.0,8352367.0,25365854.0,23017242.0,5631296.0,14000000.0,9249960.0,8809560.0,4536840.0,7983000.0,1891857.0,11245680.0,43827586.0,9900000.0,2162606.0,15500000.0,19500000.0,4454880.0,2087519.0,27556817.0,16500000.0,13394160.0,3000000.0,3193200.0,2965920.0,30000000.0,11950000.0,24107143.0,5756880.0,4231800.0,5000000.0,9057971.0,22265280.0,6836400.0,15000000.0,8000000.0,2087519.0],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"High","orientation":"v","showlegend":true,"x":[150,160,155,157,198,195,162,249,99,257,189,158,209,162,157,163,168,229,172,164,141,93,175,98,113,171,170,162,189,183,105,127,175,242,201,171,122,138,225,188,193,167,186,251,180,230,146,182,181,209,131,176,190,164,150,208,152,171,145,130,171,199,219,182,111,141,150,106,189,183,154,154],"xaxis":"x","y":[450,892,798,623,305,425,456,920,546,417,540,369,786,749,540,735,1010,972,358,418,659,481,487,590,502,522,410,383,480,471,506,798,449,275,737,401,566,432,470,370,660,564,507,807,368,450,435,785,627,297,442,547,341,395,515,293,388,440,601,707,611,492,414,548,473,466,463,469,527,399,462,455],"yaxis":"y","type":"scatter"},{"customdata":[["Trae Young"],["Tyler Herro"],["Zach LaVine"],["Jalen Green"],["Stephen Curry"],["DeMar DeRozan"],["Donovan Mitchell"],["Jalen Brunson"],["Kevin Durant"],["Darius Garland"],["Coby White"],["Austin Reaves"],["De'Aaron Fox"],["Franz Wagner"],["Damian Lillard"],["Mikal Bridges"],["Jamal Murray"],["Luka Dončić"],["Jaylen Brown"],["Jordan Poole"],["Tyrese Maxey"],["Tyrese Haliburton"],["Anfernee Simons"],["Malik Beasley"],["Shaedon Sharpe"],["Norman Powell"],["Derrick White"],["Kyrie Irving"],["RJ Barrett"],["Paolo Banchero"],["Stephon Castle"],["LaMelo Ball"],["CJ McCollum"],["Ja Morant"],["Collin Sexton"],["Payton Pritchard"],["Keyonte George"],["Trey Murphy III"],["Malik Monk"],["Quentin Grimes"],["De'Andre Hunter"],["Andrew Wiggins"],["Cameron Johnson"],["Devin Vassell"],["Harrison Barnes"],["Klay Thompson"],["Dennis Schröder"],["Keldon Johnson"],["Jimmy Butler"],["Kyle Kuzma"],["Zaccharie Risacher"],["Naji Marshall"],["Aaron Wiggins"],["Buddy Hield"],["Bradley Beal"],["Lauri Markkanen"],["Ty Jerome"],["Spencer Dinwiddie"],["Tim Hardaway Jr."],["Fred VanVleet"],["Scoot Henderson"],["Obi Toppin"],["Gary Trent Jr."],["Jaylen Wells"],["Kristaps Porziņģis"],["Duncan Robinson"],["Julian Champagnie"],["Santi Aldama"],["Georges Niang"],["Derrick Jones Jr."],["Scotty Pippen Jr."],["Gradey Dick"],["Caris LeVert"],["Rui Hachimura"],["Nickeil Alexander-Walker"],["Brice Sensabaugh"],["Kevin Porter Jr."],["John Collins"],["Isaiah Joe"],["Jalen Wilson"],["Max Christie"],["Aaron Gordon"],["Brandin Podziemski"],["Zion Williamson"],["Anthony Black"],["D'Angelo Russell"],["Bilal Coulibaly"],["Moses Moody"],["Donte DiVincenzo"],["Chris Paul"],["T.J. McConnell"],["Jonathan Kuminga"],["Dalton Knecht"],["Corey Kispert"],["Jabari Smith Jr."],["Amir Coffey"],["Matas Buzelis"],["Tari Eason"],["Jrue Holiday"],["Kevin Huerter"],["Grayson Allen"],["Jalen Johnson"],["Royce O'Neale"],["Bobby Portis"],["Terry Rozier"],["Jerami Grant"],["Kentavious Caldwell-Pope"],["Mark Williams"],["Ochai Agbaji"],["Keon Ellis"],["Paul George"],["Taurean Prince"],["Andrew Nembhard"],["Ziaire Williams"],["Cole Anthony"],["Jeremy Sochan"],["Isaiah Collier"],["Miles McBride"],["Sam Hauser"],["Ausar Thompson"],["Kyshawn George"],["Julian Strawther"],["Davion Mitchell"],["Bogdan Bogdanović"],["Mike Conley"],["Jose Alvarado"],["Deandre Ayton"],["Johnny Juzang"],["Jaime Jaquez Jr."],["Cason Wallace"],["Jalen Suggs"],["Ayo Dosunmu"],["Patrick Williams"],["Peyton Watson"],["Dorian Finney-Smith"],["Justin Champagnie"],["A.J. Green"],["Aaron Nesmith"],["Jared Butler"],["Al Horford"],["Brandon Clarke"],["Tristan Da Silva"],["Jamal Shead"],["Ricky Council IV"],["Jalen Smith"],["Tyrese Martin"],["Ron Holland"],["Terance Mann"],["Ryan Dunn"],["Sam Merrill"],["Josh Green"],["Chris Boucher"],["Cameron Payne"],["Nikola Jović"],["Chet Holmgren"],["Haywood Highsmith"],["Kris Dunn"],["Max Strus"],["Jarace Walker"],["Gabe Vincent"],["Jake LaRavia"],["Jusuf Nurkić"],["Trey Lyles"],["Trendon Watford"],["Ja'Kobe Walter"],["Simone Fontecchio"],["Justin Edwards"],["Luke Kornet"],["Kenrich Williams"],["Thomas Bryant"],["Noah Clowney"],["Jamison Battle"],["Jeremiah Robinson-Earl"],["Vít Krejčí"],["Trayce Jackson-Davis"],["Zach Collins"],["Moussa Diabaté"],["Gary Payton II"],["Day'Ron Sharpe"],["Jonathan Mogbo"],["Kelly Olynyk"],["Jaxson Hayes"],["Alex Caruso"],["Jonathan Isaac"],["Precious Achiuwa"],["Julian Phillips"],["Cody Martin"],["Kyle Anderson"],["Caleb Martin"],["Tidjane Salaün"],["Ryan Rollins"],["Javonte Green"],["Quinten Post"],["Adem Bona"],["Isaac Okoro"],["Ben Sheppard"],["Dalen Terry"],["Karlo Matković"],["Dean Wade"],["DaQuan Jeffries"],["Dereck Lively II"],["Nicolas Batum"],["Jabari Walker"],["Neemias Queta"],["Orlando Robinson"],["Andre Drummond"],["Drew Eubanks"],["Jaylin Williams"],["KJ Martin"],["Ben Simmons"],["Oso Ighodaro"],["Pelle Larsson"],["Andre Jackson Jr."],["Cody Williams"],["Gui Santos"],["Steven Adams"],["DeAndre Jordan"]],"hovertemplate":"Performance Tier=Medium\u003cbr\u003eTotal Points=%{x}\u003cbr\u003eTotal Assists=%{y}\u003cbr\u003ePlayer=%{customdata[0]}\u003cbr\u003eSalary ($)=%{marker.color}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"Medium","marker":{"color":[43031940.0,29000000.0,44531940.0,12483048.0,55761216.0,23400000.0,35410310.0,24960001.0,51179021.0,36725670.0,12000000.0,12976362.0,34848340.0,7007092.0,48787676.0,23300000.0,36016200.0,43031940.0,49205800.0,29651786.0,35147000.0,42176400.0,25892857.0,6000000.0,6614160.0,19241379.0,20071429.0,41000000.0,25794643.0,12160800.0,9105120.0,35147000.0,33333333.0,36725670.0,18350000.0,6696429.0,4084200.0,5159854.0,17405203.0,4296682.0,21696429.0,26276786.0,22500000.0,29347826.0,18000000.0,15873016.0,13025250.0,19000000.0,48798677.0,24456061.0,12569040.0,8571429.0,10514017.0,8780488.0,50203930.0,42176400.0,2560975.0,2087519.0,16193183.0,42846615.0,10259160.0,12975000.0,2087519.0,1157153.0,29268293.0,19406000.0,3000000.0,3960531.0,8500000.0,9523810.0,2087519.0,4763760.0,16615384.0,17000000.0,4312500.0,2571480.0,3366937.0,26580000.0,12991650.0,1891857.0,7142857.0,22841455.0,3519960.0,36725670.0,7607760.0,18692307.0,6945240.0,5803269.0,11445000.0,10460000.0,9300000.0,7636307.0,3819120.0,5705887.0,9770880.0,3938271.0,5195520.0,3695160.0,30000000.0,16830357.0,15625000.0,4510905.0,9375000.0,12578286.0,24924126.0,29793104.0,22757000.0,4094280.0,4310280.0,2120693.0,49205800.0,2087519.0,2019699.0,6133005.0,12900000.0,5570040.0,2512680.0,4710144.0,2092344.0,8376000.0,2825520.0,2552520.0,6451077.0,17260000.0,9975962.0,1988598.0,34005126.0,3087519.0,3685800.0,5555880.0,9188385.0,7000000.0,18000000.0,2413560.0,14924167.0,1800000.0,2120693.0,11000000.0,745726.0,9500000.0,12500000.0,3628440.0,1862265.0,1891857.0,8571429.0,635853.0,8245320.0,11423077.0,2530800.0,2164993.0,12654321.0,10810000.0,2087519.0,2464200.0,10880640.0,5200000.0,5168000.0,15212068.0,6362520.0,11000000.0,3352680.0,18125000.0,8000000.0,2726603.0,3465000.0,7692308.0,425619.0,2087519.0,6669000.0,2087519.0,3244080.0,1000000.0,2196970.0,2162606.0,1891857.0,16741200.0,957763.0,9130000.0,3989122.0,1862265.0,12804878.0,2463946.0,9890000.0,25000000.0,6000000.0,1891857.0,8120000.0,8780488.0,9186594.0,7488720.0,1091887.0,2328350.0,438920.0,1157153.0,10185186.0,2663880.0,3510480.0,1407153.0,6166667.0,2425404.0,5014560.0,4668000.0,2019699.0,2162606.0,1691610.0,5000000.0,5000000.0,2019699.0,7975000.0,40011909.0,1157153.0,1157153.0,1891857.0,5469120.0,1891857.0,12600000.0,2087519.0],"coloraxis":"coloraxis","symbol":"diamond"},"mode":"markers","name":"Medium","orientation":"v","showlegend":true,"x":[145,85,121,127,95,150,140,138,104,143,154,151,164,147,97,127,128,126,154,201,116,92,117,125,120,114,138,101,137,97,164,157,124,113,144,118,111,112,158,138,155,101,104,108,75,81,180,115,52,119,149,109,101,132,136,79,102,121,96,140,181,108,124,146,116,135,114,76,201,153,203,120,99,105,139,111,119,116,113,164,115,82,95,82,166,115,139,116,114,151,79,90,93,98,123,120,128,136,101,121,95,67,151,94,73,109,136,105,125,176,101,165,149,149,151,131,149,108,87,164,201,151,185,127,111,91,88,126,81,145,98,105,88,123,136,110,157,114,97,81,145,94,158,67,95,77,141,128,162,124,165,68,111,82,72,164,197,104,115,152,132,130,86,95,95,95,98,119,110,76,105,92,79,105,73,132,109,108,113,136,123,136,102,90,81,108,95,81,87,84,88,98,83,125,104,125,114,106,101,99,100,111,87,106,100,94,63,79,101,107,102,91,114,95,91,60,74],"xaxis":"x","y":[236,399,315,377,310,298,322,187,374,214,273,329,298,342,272,259,261,409,368,203,174,258,189,214,324,190,341,240,366,345,297,232,212,205,171,307,253,268,245,324,258,269,247,255,310,246,198,371,296,370,268,328,295,264,177,278,173,209,183,221,198,318,168,268,284,167,318,416,272,263,259,193,203,295,265,210,278,327,196,270,258,247,325,216,230,162,294,189,228,296,193,217,216,181,399,161,278,362,265,205,194,359,349,411,238,163,169,447,242,212,219,287,216,287,204,352,235,161,224,303,285,142,167,159,182,136,406,184,290,229,141,161,237,233,247,351,174,178,110,369,326,245,115,214,357,219,218,199,267,157,172,224,104,180,257,252,251,217,229,92,259,398,316,160,160,215,148,388,241,251,181,158,315,156,311,288,438,185,330,311,206,269,159,314,317,169,224,206,177,280,108,218,148,245,130,177,124,209,249,135,270,218,211,236,220,310,226,262,125,241,222,92,182,113,172,327,284],"yaxis":"y","type":"scatter"},{"customdata":[["Tyus Jones"],["Kawhi Leonard"],["Jordan Hawkins"],["Jordan Clarkson"],["Cam Thomas"],["Nick Smith Jr."],["Luke Kennard"],["Brandon Miller"],["Immanuel Quickley"],["Dalano Banton"],["Dejounte Murray"],["Jaden Ivey"],["Jaden Hardy"],["Cam Whitmore"],["Joel Embiid"],["Brandon Boston Jr."],["Seth Curry"],["Khris Middleton"],["Jay Huff"],["Brandon Ingram"],["Moritz Wagner"],["Sandro Mamukelashvili"],["Talen Horton-Tucker"],["Marcus Sasser"],["Alec Burks"],["Garrison Mathews"],["Jared McCain"],["Aaron Holiday"],["Bruce Brown"],["Svi Mykhailiuk"],["Tre Jones"],["Shake Milton"],["Malcolm Brogdon"],["Antonio Reeves"],["Marcus Smart"],["Kris Murray"],["Landry Shamet"],["Josh Okogie"],["Brandon Williams"],["Vasilije Micić"],["Lonzo Ball"],["Jett Howard"],["Eric Gordon"],["Lindy Waters III"],["Lonnie Walker IV"],["Bol Bol"],["Caleb Houstan"],["A.J. Lawson"],["Malaki Branham"],["Ajay Mitchell"],["Monte Morris"],["Richaun Holmes"],["Reed Sheppard"],["AJ Johnson"],["Rob Dillingham"],["Blake Wesley"],["Pat Connaughton"],["GG Jackson II"],["Herbert Jones"],["Larry Nance Jr."],["Olivier-Maxence Prosper"],["Jock Landale"],["Jalen Pickett"],["Mouhamed Gueye"],["Duop Reath"],["Craig Porter Jr."],["Jae'Sean Tate"],["Zeke Nnaji"],["Paul Reed"],["Tre Mann"],["Vince Williams Jr."],["Cory Joseph"],["Danté Exum"],["Jeff Green"],["Jaylon Tyson"],["Grant Williams"],["Jaylen Clark"],["Daniel Theis"],["Jordan Goodwin"],["Charles Bassey"],["Colby Jones"],["Rayan Rupert"],["Jevon Carter"],["David Roddy"],["Jordan Miller"],["Wendell Moore Jr."],["Bones Hyland"],["Dominick Barlow"],["Doug McDermott"],["Jarred Vanderbilt"],["Gary Harris"],["Bismack Biyombo"],["Ousmane Dieng"],["Devin Carter"],["Luka Garza"],["Kyle Lowry"],["Terrence Shannon Jr."],["Mo Bamba"],["Reggie Jackson"],["Marvin Bagley III"],["Dillon Jones"],["Isaac Jones"],["Hunter Tyson"],["Anthony Gill"],["Delon Wright"],["Colin Castleton"],["Branden Carlson"],["Kevin Love"],["Josh Minott"],["Jared Rhoden"],["Robert Williams"],["Maxwell Lewis"],["Dariq Whitehead"],["Dwight Powell"],["Baylor Scheierman"],["John Konchar"],["Matisse Thybulle"],["Patty Mills"],["Drew Timme"],["Jaden Springer"],["Torrey Craig"],["Taj Gibson"],["Jalen Hood-Schifino"],["Cam Reddish"],["Johnny Furphy"],["Maxi Kleber"],["Pat Spencer"],["Jericho Sims"],["Jordan McLaughlin"],["Mitchell Robinson"],["Elfrid Payton"],["Damion Lee"],["Jordan Walsh"],["Tyler Kolek"],["Johnny Davis"],["MarJon Beauchamp"],["Kobe Brown"],["Moses Brown"],["Lamar Stevens"],["Alex Len"],["Tristan Thompson"],["Marcus Bagley"],["Jaylen Nowell"],["Tyler Smith"],["Bronny James"],["De'Anthony Melton"],["Tony Bradley"],["Markelle Fultz"],["Cole Swider"],["Bruno Fernando"],["Dario Šarić"],["Markieff Morris"],["Killian Hayes"],["Kobe Bufkin"],["Garrett Temple"],["Chuma Okeke"],["Patrick Baldwin Jr."],["Oshae Brissett"],["Ariel Hukporti"],["Kevin Knox"],["Keshad Johnson"],["Jaylen Sims"],["Chris Duarte"],["Nae'Qwan Tomlin"],["Isaiah Jackson"],["Xavier Tillman Sr."],["Kylor Kelley"],["JD Davison"],["Josh Richardson"],["Sidy Cissoko"],["Pacôme Dadiet"],["Chris Livingston"],["Marcus Garrett"],["Matt Ryan"],["Vlatko Čančar"],["Jae Crowder"],["E.J. Liddell"],["Leonard Miller"],["Cam Christie"],["Malachi Flynn"],["Joe Ingles"],["Bobi Klintman"],["Taylor Hendricks"],["P.J. Tucker"],["James Johnson"],["Phillip Wheeler"],["PJ Dozier"],["Isaiah Mobley"],["James Wiseman"],["Skal Labissière"],["Braxton Key"],["Yuri Collins"],["Malevy Leons"],["Daishen Nix"],["Terence Davis"],["Jalen McDaniels"],["Jahlil Okafor"],["Terry Taylor"]],"hovertemplate":"Performance Tier=Low\u003cbr\u003eTotal Points=%{x}\u003cbr\u003eTotal Assists=%{y}\u003cbr\u003ePlayer=%{customdata[0]}\u003cbr\u003eSalary ($)=%{marker.color}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"Low","marker":{"color":[2087519.0,49205800.0,4525680.0,14092577.0,4041249.0,2587200.0,9250000.0,11424600.0,32500000.0,2196970.0,29517135.0,7977240.0,2019699.0,3379080.0,51415938.0,596581.0,2087519.0,31000000.0,2088033.0,36016200.0,11000000.0,2087519.0,2087519.0,2755080.0,2087519.0,2230253.0,4020360.0,4668000.0,23000000.0,3500000.0,9104167.0,2875000.0,22500000.0,1157153.0,20210284.0,2990040.0,1343690.0,8250000.0,47989.0,7723000.0,21395348.0,5278320.0,3303771.0,2196970.0,780932.0,2087519.0,2019699.0,100000.0,3217920.0,3000000.0,2087519.0,12648321.0,10098960.0,2795294.0,6262920.0,2624280.0,9423869.0,1891857.0,12976362.0,11205000.0,2870400.0,8000000.0,1891857.0,1891857.0,2048780.0,1891857.0,7565217.0,8888889.0,3913234.0,4908373.0,2120693.0,3303771.0,3150000.0,8000000.0,3326160.0,13025250.0,492323.0,2087519.0,223718.0,2087519.0,2120693.0,1891857.0,6500000.0,2967212.0,1050000.0,2537040.0,4158439.0,491887.0,2087519.0,10714286.0,7500000.0,647851.0,5027040.0,4689000.0,2162606.0,2087519.0,2546640.0,2207491.0,4033748.0,12500000.0,2622360.0,152957.0,1891857.0,2237691.0,2087519.0,421081.0,496519.0,3850000.0,2019699.0,119972.0,12428571.0,1891857.0,3114240.0,4000000.0,2494320.0,6165000.0,11025000.0,2087519.0,113055.0,4772772.0,3625162.0,2087519.0,3879840.0,2463946.0,1850842.0,11000000.0,438810.0,2092344.0,2087519.0,14318182.0,789048.0,2087519.0,1891857.0,2087519.0,5291160.0,2733720.0,2533920.0,426632.0,623856.0,2831348.0,2087519.0,126356.0,446743.0,1157153.0,1157153.0,12822000.0,600545.0,731831.0,227947.0,1115128.0,5168000.0,2087519.0,119972.0,4299000.0,2087519.0,355687.0,2448840.0,119972.0,1064049.0,503883.0,724883.0,73153.0,5893768.0,66503.0,4435381.0,2237691.0,73153.0,11997.0,3051153.0,1891857.0,1808080.0,1891857.0,107027.0,621439.0,2087519.0,1655619.0,706898.0,1891857.0,1157153.0,119972.0,2087519.0,1257153.0,5848680.0,12025777.0,2087519.0,66503.0,1051255.0,11997.0,2237691.0,119972.0,11997.0,66503.0,126356.0,119972.0,64301.0,4861772.0,119972.0,119972.0],"coloraxis":"coloraxis","symbol":"square"},"mode":"markers","name":"Low","orientation":"v","showlegend":true,"x":[63,56,51,52,44,79,59,74,61,58,61,67,90,44,42,73,59,75,78,45,55,60,59,72,37,81,37,64,71,43,55,67,30,40,53,66,52,47,33,45,55,68,32,62,27,20,62,44,29,68,23,49,53,50,37,46,45,30,66,37,28,49,38,55,41,25,82,62,75,24,45,50,34,32,50,46,61,70,43,50,37,37,15,20,26,31,29,47,30,67,45,52,33,39,29,62,27,56,34,18,54,30,44,30,25,51,22,21,34,20,31,29,22,75,22,32,14,24,21,42,34,54,25,43,41,66,20,65,18,26,36,6,30,26,18,14,23,17,15,56,36,16,9,11,13,18,11,12,12,27,19,16,13,9,23,5,7,14,32,6,11,10,7,9,14,18,15,2,17,10,7,8,10,1,10,10,11,2,4,4,12,8,8,8,5,1,3,3,0,1,0,5,4,1,0,0,0,0],"xaxis":"x","y":[196,219,158,118,83,128,183,131,117,135,200,124,89,151,155,134,113,137,129,100,148,187,100,69,124,87,56,78,164,90,116,104,91,63,72,182,61,113,58,89,118,70,46,110,63,104,75,86,51,68,67,177,78,59,50,61,109,92,78,103,126,137,71,139,92,67,117,90,123,38,98,74,33,58,96,82,53,164,112,152,87,70,39,84,58,95,25,85,22,182,64,158,81,76,54,67,47,145,42,84,121,57,78,66,66,122,54,94,47,42,118,55,29,118,65,151,52,22,65,58,73,120,27,66,71,95,46,197,30,101,79,20,69,27,38,44,65,44,38,95,134,70,20,25,18,20,42,20,27,51,50,23,18,21,29,47,25,22,51,17,28,6,20,21,28,43,39,13,12,21,18,35,6,8,32,20,9,11,12,7,11,7,15,8,6,8,5,4,1,3,2,3,3,4,1,0,1,1],"yaxis":"y","type":"scatter"},{"hoverinfo":"skip","marker":{"color":"red","line":{"color":"black","width":1},"size":16,"symbol":"star"},"mode":"markers+text","name":"Top Targets","text":["Jalen Duren","Alperen Şengün","Zach Edey","Jonas Valančiūnas","Onyeka Okongwu","Josh Hart"],"textfont":{"family":"Arial","size":10},"textposition":"top center","x":[251,209,182,181,193,201],"y":[807,786,548,627,660,737],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Total Points"},"gridcolor":"white","showgrid":true},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Total Assists"},"gridcolor":"white","showgrid":true},"coloraxis":{"colorbar":{"title":{"text":"Salary ($)"},"tickformat":"~s","thickness":15,"len":0.7,"x":1.02},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"legend":{"title":{"text":"Tier \u002f Highlight"},"tracegroupgap":0,"x":1.15,"y":0.8},"title":{"text":"The Value Gap: NBA Player Production vs. Salary (2024–25)","font":{"size":20,"family":"Arial"},"x":0.5,"xanchor":"center"},"font":{"family":"Arial","size":12},"margin":{"l":50,"r":200,"t":90,"b":60},"plot_bgcolor":"rgba(245, 245, 245, 1)","paper_bgcolor":"white","annotations":[{"font":{"color":"gray","family":"Arial","size":10},"showarrow":false,"text":"Source: Basketball Reference 2024–25 | ★ = Top Value Targets","x":0,"xref":"paper","y":-0.12,"yref":"paper"}]},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4de15b87-5964-4202-bc2a-2d24f0abd32c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

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
