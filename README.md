# DS 4320 Project 2: The Value Gap
### Using K-Means Clustering to Expose Salary Inefficiencies in the 2024–25 NBA Season

**Executive Summary:** 

This project, *The Value Gap*, investigates salary inefficiencies in the 2024–25 NBA season using k-means clustering applied to player performance data sourced from Basketball Reference. Two datasets — a player totals table and a salary table — were cleaned, merged, and loaded into MongoDB Atlas, then analyzed through a Python pipeline using scikit-learn. Players are clustered into three performance tiers (High, Medium, Low) based on total points and assists, with salary overlaid to identify who is being paid fairly, who is underpaid, and who is overpaid relative to their output. The primary deliverable is an actionable shortlist of signing targets and players to avoid, framed for a non-technical front office audience through a press release and visualization. Known limitations include a position bias toward ball-handlers, survivorship bias from injured players sitting out the full season, and the inherent uncertainty of using next-year contract value as a proxy for current player worth. Full methodology, metadata, and bias documentation are included in this repository.

**Name:** Avalon Bennett

**NetID:** dqb5ee

**DOI:** 

**Press Release:** [View "The Value Gap" Press Release]([link](https://github.com/dqb5ee/project-2-dqb5ee/blob/main/pressrelease.md))

**Data:** [OneDrive Folder](https://myuva-my.sharepoint.com/:b:/g/personal/dqb5ee_virginia_edu/IQAeCwx7j_UlR7sSh6Bc4wUZAfx7X2BF973VGb6OXb_ep9I?e=c2EqMA)

**Pipeline:** [Technical Pipeline & Execution Log]([link](https://github.com/dqb5ee/project-2-dqb5ee/tree/main/pipeline))

**License:** This project is licensed under the [MIT License.]([link](https://github.com/dqb5ee/project-2-dqb5ee/blob/main/license.md))

## Problem Definition

**General Problem:** Can we project athlete performance and salary based on points and assists?

**Refined Statement:** 

Using 2024–2025 NBA season data, can we cluster on points (PTS) and assists (AST) to identify which players are statistically underpaid relative to their performance tier, and which should be avoided due to high salary and low output?

**Rationale for Refinement:** 

The NBA salary cap creates a zero-sum environment: every dollar spent on an
underperforming player is a dollar that cannot be used to sign someone better. Front offices often rely on reputation, draft pedigree, or prior-contracts when making signing decisions, rather than current performance data.

**Motivation:** 

This project is motivated by the belief that a data-driven, unsupervised clustering approach can cut through those biases and surface players whose market value is misaligned with their actual output. Identifying even one or two underpaid high-performers (or avoiding one costly low-performer) could meaningfully shift a team's competitive standing within the salary cap constraints of the 2024–2025 season.

**Press Release Headline:** The NBA's Best-Kept Secret: High Performers, Low Paychecks

[**Press Release Link**](link)


## Domain Exposition

**Terminology:**

| Term | Definition |
|:--|:--|
| **PTS (Points)** | Total points scored by a player per game in the 2024–25 season |
| **AST (Assists)** | Total assists (passes leading to a score) per game |
| **Salary Cap** | The maximum total salary an NBA team can pay its players in a season |
| **K-Means Clustering** | An unsupervised ML algorithm that groups data points into k clusters based on similarity |
| **Cluster** | A group of players with similar performance profiles (High, Medium, Low) |
| **Silhouette Score** | A metric from -1 to 1 measuring how well-separated clusters are; higher is better |
| **Underpaid Player** | A player in the High performance cluster whose salary falls in the bottom third of all salaries |
| **Overpaid Player** | A player in the Medium or Low cluster whose salary falls in the top third |
| **eFG% (Effective Field Goal %)** | A shooting efficiency metric that accounts for the extra value of three-pointers |
| **2024–2025 Season** | The NBA season used as the basis for this analysis, looked at previous season in order to get a full year of results |

**Project Domain:** 

This project lives at the intersection of sports analytics and unsupervised machine learning. Sports analytics has grown into a multi-billion-dollar industry since the early 2000s, when NBA franchises began using statistical modeling to find undervalued talent. In the NBA specifically, the salary cap means that the way a roster is constructed is as much a financial puzzle as it is a talent challenge. Teams that can consistently identify high-value players (those who produce at an elite level but are paid below market rate - often due to age, tenure, or simply being overlooked) gain a structural competitive advantage. This project applies k-means clustering to player performance and compensation data from the 2024–2025 season to surface those mismatches programmatically and provide actionable recruiting recommendations.

**Summary of Background Research:**


| # | Title | Description | Link |
|:--|:--|:--|:--|
| 1 | Moneyball And The Rise Of Sports Analytics | Background on how data-driven decision making entered professional sports front offices. |  [Link to Article](https://myuva-my.sharepoint.com/:b:/g/personal/dqb5ee_virginia_edu/IQDEHalWww-5TJP58RISXWRHATTV0ty9FDUF3RaPeNfvSNs?e=WEafGt) |
| 2 | Basketball analytics investment is key to NBA wins and other successes | How modern NBA franchises use ML for player evaluation. | [Link to Article](https://myuva-my.sharepoint.com/:b:/g/personal/dqb5ee_virginia_edu/IQAeCwx7j_UlR7sSh6Bc4wUZAfx7X2BF973VGb6OXb_ep9I?e=xV3fJH) |
| 3 | Understanding K-means Clustering: A Comprehensive Guide. | A primer on the algorithm used in this project, including how to choose k and interpret silhouette scores. | [Link to Article](https://myuva-my.sharepoint.com/:b:/g/personal/dqb5ee_virginia_edu/IQBuaBUxCbsARbhS2iGU7-vkAZkMQBTXU9Bm-3kVXCZvLU8?e=gRGfcV) |
| 4 | How Does the NBA Salary Cap Work? | NBA Salary Cap explained and why it matters for roster construction decisions. | [Link to Article](https://myuva-my.sharepoint.com/:b:/g/personal/dqb5ee_virginia_edu/IQA14-X7PGdLSKo0X_S9QIpOAbO6UNyVvky9zFSvxOz_DtY?e=meKqtl) |
| 5 | Data reveals the value of an assist in basketball | Why assists are an underrated metric for evaluating player impact on winning. | [Link to Article](https://myuva-my.sharepoint.com/:b:/g/personal/dqb5ee_virginia_edu/IQC2knKsrLheSInnUt82IOjuATOrn2q2hfuczyIuusJlzFw?e=4XnyLV) |

## Data Creation

**Provenance:** 


The data for this project was sourced entirely from
[Basketball Reference](https://www.basketball-reference.com), the most
comprehensive publicly available repository of NBA statistics. Two separate
tables were manually exported from the site for the 2024–25 season, the first being a `player totals table` containing cumulative season statistics (points,
assists, rebounds, shooting percentages, games played, etc.) and the second was a `player salary table` containing each player's current and future contracted salary by season. Both tables were downloaded directly from Basketball Reference's export functionality, which let me copy a CSV-formatted output

Because Basketball Reference lists players who were traded midseason multiple times (once per team and once as a combined total) the totals rows which were flagged as `2TM` or `3TM` in the `Tm` column were retained and all individual team rows for those players were dropped during preprocessing, using a deduplication step that keeps only the first occurrence of each player name. The salary data was similarly deduplicated to one row per player before merging. The cleaned and merged dataset was then used directly in the clustering analysis without being loaded into a database yet, although the HW9 notebook demonstrates the same workflow applied through a MongoDB connection.

**Code:**

| File | Description | Link to Code |
|:--|:--|:--|
| `NBA_Totals_2425.csv` | Raw 2024-25 season cumulative player stats exported from Basketball Reference | [Link](https://github.com/dqb5ee/project-2-dqb5ee/blob/main/code/data_creation_code.py) |
| `NBA_Salaries_2425.csv` | Player contract salary data for 2024-25 and future seasons from Basketball Reference | [Link](https://github.com/dqb5ee/project-2-dqb5ee/blob/main/code/data_creation_code.py) |

**Bias Identification:** 

Several sources of bias could be introduced during the data collection process for this project. First, survivorship bias is present because Basketball Reference only lists players who appeared in at least one game during the 2024–25 season, so injured players who never suited up are excluded entirely, which skews the salary distribution upward since high-earning players on long-term contracts (such as Joel Embiid, who missed most of the season) are included in the salary file but have limited or missing performance data. Second, recency bias is found in the salary column because contract values reflect what a player was worth at the time they were signed, not their current performance level. A player locked into a max contract three years ago based on past performance may now be declining, but their salary still appears at the top of the distribution. Lastly, position bias is introduced by using only PTS and AST as clustering features because these metrics naturally favor guards and forwards who handle the ball, which systematically under-represents centers and defensive players whose value is not captured in these two columns.

**Bias Mitigation:** 

While some biases in this dataset cannot be fully eliminated, several steps can be taken to reduce their impact. Survivorship bias can be partially addressed by flagging and separately analyzing players with high salaries but low game counts (like filtering for a minimum games played threshold before clustering, so injured high earners don't distort the salary tiers). Recency bias in salaries can be quantified by calculating each player's salary relative to their performance percentile and flagging outliers (players
whose salary rank deviates significantly from their performance rank) rather than treating salary as a ground-truth measure of value. Position bias can be mitigated by running separate clustering analyses per position group, or by expanding the feature set to include metrics that better capture defensive and rebounding value (e.g., BLK, TRB, STL) before clustering. In the current analysis, the bias is acknowledged and the results are framed explicitly as an offensive value analysis rather than total player value, which is an important caveat for any front office using these findings.

**Rationale:** 

Several judgment calls were made during this project that introduce or mitigate uncertainty and need further explanation.

First, the decision to use three clusters (High, Medium, Low) was made to keep the results interpretable for a non-technical audience. A higher k would produce finer grained groupings that would make the visualization and the signing recommendations harder to communicate. The
silhouette score is printed at runtime to validate that three clusters produce meaningful separation for the given season's data, because if it falls
below 0.3 then k should be reconsidered.

Season totals were chosen over per-game averages because the goal is to identify players who contributed the most absolute output to their team, which is more relevant
to roster-building than efficiency per appearance. A player who averages 25 points but only played 20 games has less of a steady total value than one averaging 18 points over 80 games. But, this decision disadvantages players who missed games due to injury and is a known limitation.

Keeping only the first row per player (the season-total row in the Basketball Reference export) makes sure
no player is counted twice. However, this means their team assignment (`Tm`) reflects their total row label (`2TM`), not the team they most recently
played for, which is a limitation for any analysis that factors in team-level context. Luckily this one does not.

The `2025-26` salary column was used rather than the current season's cap hit because this represents the player's next contract obligation, which is more relevant for a front office making forward-looking signing decisions. This introduces uncertainty for players on expiring deals or those with team/player options, whose actual 2025-26 salary may differ from what is listed.

## Metadata

**Implicit Schema:** 

The merged dataset follows a flat, row-per-player structure where each
record represents one NBA player's complete 2024–25 season. The implicit
schema conventions are as follows:

- One row per player: Multi-team rows are collapsed to a single totals row. No player appears more than once
- Numeric performance fields (PTS, AST, TRB, etc.) store integer or float values representing season cumulative totals
- Percentage fields (FG%, 3P%, FT%, eFG%) store float values between 0.0 and 1.0 (not 0–100)
- Salary is stored as a float in US dollars (e.g., `48070014.0`), with `$` and `,` stripped during preprocessing
- Categorical fields (Player, Pos, Tm, cluster) store strings
- Missing values arise legitimately (e.g., a player with zero three-point attempts has no 3P%  and is stored as `NaN`, not zero) and should not be imputed without domain justification
- The `cluster` column is a derived field added during analysis (not present in the raw source files) and takes values `'High'`, `'Medium'`, or `'Low'`


**Data Summary:**

| Property | Value |
|:--|:--|
| Source | Basketball Reference (basketball-reference.com) |
| Season | 2024–25 NBA Regular Season |
| Files | 2 (NBA_Totals_2425.csv, NBA_Salaries_2425.csv) |
| Players in performance file | ~570 rows (pre-deduplication) |
| Players after deduplication | ~450 unique players |
| Players after salary merge | ~400 (inner join, salary required) |
| Performance features | 29 columns (statistics and identifiers) |
| Salary features | 6 columns (current and future seasons) |
| Derived features | 1 (`cluster`: High / Medium / Low) |
| Missing values | Present in percentage columns for players with 0 attempts |
| Date of export | April 2026 (to gather a full season of data) |

**Data Dictionary:**

| Feature | Type | Description | Example |
|:--|:--|:--|:--|
| `Player` | string | Full player name | `"Shai Gilgeous-Alexander"` |
| `Age` | int | Player age at start of season | `26` |
| `Tm` | string | Team abbreviation (or `2TM`/`3TM` for multi-team) | `"OKC"` |
| `Pos` | string | Primary position | `"PG"` |
| `G` | int | Games played | `76` |
| `GS` | int | Games started | `76` |
| `MP` | int | Total minutes played | `2598` |
| `FG` | int | Field goals made | `860` |
| `FGA` | int | Field goal attempts | `1656` |
| `FG%` | float | Field goal percentage (0.0–1.0) | `0.519` |
| `3P` | int | Three-point field goals made | `163` |
| `3PA` | int | Three-point attempts | `435` |
| `3P%` | float | Three-point percentage (0.0–1.0) | `0.375` |
| `2P` | int | Two-point field goals made | `697` |
| `2PA` | int | Two-point attempts | `1221` |
| `2P%` | float | Two-point percentage (0.0–1.0) | `0.571` |
| `eFG%` | float | Effective field goal % (adjusts for 3P value) | `0.569` |
| `FT` | int | Free throws made | `601` |
| `FTA` | int | Free throw attempts | `669` |
| `FT%` | float | Free throw percentage (0.0–1.0) | `0.898` |
| `ORB` | int | Offensive rebounds | `67` |
| `DRB` | int | Defensive rebounds | `312` |
| `TRB` | int | Total rebounds | `379` |
| `AST` | int | Total assists | `486` |
| `STL` | int | Total steals | `131` |
| `BLK` | int | Total blocks | `77` |
| `TOV` | int | Total turnovers | `183` |
| `PF` | int | Personal fouls | `164` |
| `PTS` | int | Total points scored | `2484` |
| `Salary` | float | 2025–26 contracted salary in USD | `48070014.0` |
| `cluster` | string | K-means performance tier (derived) | `"High"` |

**Quantification of Uncertainty:**

| Feature | Uncertainty Source | Quantification | Notes |
|:--|:--|:--|:--|
| `PTS` | Games played varies (20–82 games) | Up to **4.1× difference** in scoring opportunities between a player with 82 games vs. 20 | Totals favor high-availability players regardless of per-game efficiency |
| `AST` | Scorer's table subjectivity | **~2–5% variability** by arena; ~±8–20 assists for a player with 400 on the season | Assist credit depends on official scorer judgment, which varies slightly by city |
| `FG%` / `3P%` / `FT%` | Small sample sizes | One make or miss swings the metric by **≥2 pts** at 50 attempts, **5 pts** at 20 attempts | Treat as unreliable for players with fewer than 50 attempts |
| `eFG%` | Derived metric | Inherits error from both `FG` and `3P`; combined uncertainty of **±3–4%** | A single shot misclassified as a 2 instead of a 3 directly shifts this figure |
| `Salary` | Contract structure complexity | True cap hit can deviate from listed value by **$1M–$10M+** for ~15–20% of the roster | Incentive clauses and partial guarantees are not reflected in the base figure |
| `cluster` | K-means algorithm sensitivity | ~**10–15% of players** fall close enough to a boundary to flip clusters under a different random seed | Boundary players should not be treated as confidently classified |
| `MP` | Load management / injury | Same `MP` total can reflect anywhere from **50 to 82 games**, a **64% difference** in context | Does not distinguish strategic rest from injury absence |
