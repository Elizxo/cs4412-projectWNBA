# cs4412-projectWNBA

# Author

**Elizabeth Serrano**

- Individual Project

# WNBA Draft Player Data Analysis (1997-2022)

## Project Description

This project analyzes historical WNBA draft and career performance data from 1997 to 2022 to uncover patterns in long-term player success and team drafting effectiveness. The goal is to explore how well draft position predicts career outcomes, determine which teams most consistently draft impactful players, and identify athletes who significantly exceeded or fell short of expectations.

Data mining techniques such as **clustering** and **anomaly detection** are used to group players by career performance characteristics and highlight significant outliers.

All visualizations and data mining techniques are implemented using **Python in Google Colab**, which is used for data preprocessing, exploratory data analysis, and machine learning techniques such as clustering.

## Dataset

The dataset used for this project is titled **WNBA Draft Player Data Analysis (1997-2022)** and is publicly available on Kaggle.

- **Source:** https://www.kaggle.com/code/mattop/wnba-draft-player-data-analysis-1997-2022

### Dataset Features

- overall_pick : Draft position  
- year : Draft year  
- team : Drafting team  
- player : Player name  
- former : Former team or league  
- college : College attended  
- years_play : Years played in the WNBA  
- games : Total career games played  
- win_share : Career win share  
- win_share_40 : Win share per 40 minutes  
- minutes_p : Average minutes per game  
- points : Average points per game  
- total_rebc : Average rebounds per game  
- assists : Average assists per game  

**Known data quality issues** include missing career statistics for players 

## Discovery Questions

- Which WNBA teams most consistently draft players who achieve successful long-term careers based on career performance metrics such as win share, games played, and years in the league?

- Which drafted players most significantly overperformed or underperformed career expectations relative to their draft position?

- Which players delivered the greatest career impact per minute played, and what patterns distinguish high-efficiency players from high-volume players?

## Planned Techniques

- **Clustering (Used for Discovery Question 3)**  
  I applied K-Means clustering to group players based on performance metrics such as points, assists, rebounds, minutes played, and win share. This technique helps identify patterns between high-impact, high-volume, and low-impact players and directly supports Discovery Question 3, which focuses on identifying players who deliver the greatest career impact relative to their playing time.

- **Anomaly Detection (Planned for Discovery Question 2 - Still Under Consideration)**  
  Anomaly detection will be used to identify players who significantly overperformed or underperformed relative to expectations based on their draft position and projected career trajectory.

- **Optional: Dimensionality Reduction (PCA)**  
  Principal Component Analysis (PCA) may be used to reduce the dimensionality of the dataset and improve visualization of player groupings and performance patterns.

## Preliminary Timeline

- Week 1 (Jan 12 to Feb 8) - M1: Project Proposal  
- Weeks 2 to 4 (Feb 9 to Mar 8) - M2: Initial Implementation  
- Weeks 5 to 7 (Mar 9 to Apr 5) - M3: Complete Implementation  
- Weeks 8 to 9 (Apr 30 to May 4) - M4: Final Deliverable  

## Anticipated Challenges

- Cleaning and preprocessing raw data (removed duplicates and rows with missing data)
- Handling missing or incomplete information  
- Normalizing and standardizing features  
- Choosing appropriate clustering parameters  
- Distinguishing true anomalies from normal variation  
