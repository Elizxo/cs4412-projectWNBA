import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from google.colab import files
# Select the File from your drive
# Upload file
uploaded = files.upload()

# Load the uploaded CSV into df
df = pd.read_csv(next(iter(uploaded)))

df = pd.read_csv("wnbadraft.csv")
display(df.head())
print("Shape:", df.shape)
df.info()
display(df.describe())

# Only removes data that did not have stats like assits, etc
# Data Cleaning
df = pd.read_csv("wnbadraft.csv")

print("Shape before cleaning:", df.shape)
print("\nMissing values before cleaning:")
print(df.isna().sum())

# Only drop rows missing actual performance stats needed for analysis
stats_cols = ["games", "win_shares", "win_shares_40", "minutes_played",
              "points", "total_rebounds", "assists"]
df = df.dropna(subset=stats_cols)

# Drop duplicates
df = df.drop_duplicates()

# Remove players with 0 minutes played
df = df[df["minutes_played"] > 0]

# Drop player rows with no name
df = df.dropna(subset=["player"])

print(f"\nClean dataset shape: {df.shape}")
print("Duplicates remaining:", df.duplicated().sum())

print("\nCleaned Data Summary Statistics")
display(df.describe())

# Exploratory Data Analysis

# Plotted histogram of career points
plt.figure(figsize=(8, 4))
df["points"].hist(bins=30, color="pink", edgecolor="white")
plt.title("Distribution of Career Points")
plt.xlabel("Points")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.show()

# Plotted box plot of win shares
plt.figure(figsize=(5, 5))
plt.boxplot(df["win_shares"], vert=True, patch_artist=True,
            boxprops=dict(facecolor="lightblue"), widths=0.5)
plt.title("Win Shares Distribution")
plt.ylabel("Win Shares")
plt.xticks([1], ["Win Shares"])
plt.tight_layout()
plt.show()

# Plotted scatter of minutes played vs win shares
plt.figure(figsize=(10, 6))
plt.scatter(df["minutes_played"], df["win_shares"], alpha=0.5, color="teal")
plt.xlabel("Minutes Played")
plt.ylabel("Win Shares")
plt.title("Minutes Played versus Win Shares")
plt.tight_layout()
plt.show()

# Plotted correlation heatmap of player stats
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Player Statistics")
plt.tight_layout()
plt.show()

# Derived feature: impact per minute
# Formula: impact_per_minute = win_shares / minutes_played
# Showed how effective a player was while on the court. High value = high efficiency
df["impact_per_minute"] = df["win_shares"] / df["minutes_played"]

# Q3 : K MEANS CLUSTERING
# Goal: find high-efficiency vs high-volume players
# Features included minutes, points, assists, total rebounds, win shares, and impact per minute
# Impact per minute formula: impact_per_minute = win_shares / minutes_played
# Shows how effective a player was while on the court. High value = high efficiency

features = [
    "minutes_played",
    "points",
    "assists",
    "total_rebounds",
    "win_shares",
    "impact_per_minute"
]

X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method was used to determine the best number of clusters
# SSE (sum of squared errors) measures cluster compactness. Lower SSE = tighter clusters
sse = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), sse, marker="o", color="steelblue")
plt.xlabel("Number of Clusters k")
plt.ylabel("SSE Inertia")
plt.title("Elbow Method for Choosing k")
plt.tight_layout()
plt.show()

# Silhouette scores were calculated to justify cluster choice
# Formula: silhouette = (b - a) / max(a, b)
# a = average distance within cluster, b = average distance to nearest other cluster
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    print(f"  k={k}  silhouette score: {score:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), sil_scores, marker="o", color="darkorange")
plt.axvline(x=3, color="red", linestyle="--", label="k=3 selected")
plt.xlabel("Number of Clusters k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores to Justify k=3")
plt.legend()
plt.tight_layout()
plt.show()

# Final KMeans clustering was performed with k=3
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Cluster averages were calculated to summarize player stats per cluster
print("\nCluster Averages:")
cluster_summary = df.groupby("cluster")[[
    "games", "years_played", "win_shares",
    "points", "minutes_played", "impact_per_minute"
]].mean().round(3)
display(cluster_summary)

# Sample players per cluster were displayed to see top performers by win shares
for c in sorted(df["cluster"].unique()):
    print(f"\nCluster {c} Sample Players")
    display(df[df["cluster"] == c][[
        "player", "points", "minutes_played",
        "win_shares", "impact_per_minute"
    ]].sort_values("win_shares", ascending=False).head(8))

# Scatter plot of minutes played vs win shares colored by cluster
# Shows which clusters contained high-volume players and high-impact players
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df["minutes_played"], df["win_shares"],
                      c=df["cluster"], cmap="Set1", alpha=0.7)
plt.xlabel("Minutes Played")
plt.ylabel("Win Shares")
plt.title("Q3 Player Clusters Volume vs Win Shares")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# Scatter plot of minutes played vs impact per minute colored by cluster
# Shows which clusters were most efficient relative to minutes played
plt.figure(figsize=(10, 6))
scatter2 = plt.scatter(df["minutes_played"], df["impact_per_minute"],
                       c=df["cluster"], cmap="Set1", alpha=0.7)
plt.xlabel("Minutes Played")
plt.ylabel("Impact Per Minute")
plt.title("Q3 Efficiency vs Volume by Cluster")
plt.colorbar(scatter2, label="Cluster")
plt.tight_layout()
plt.show()

# DECISION TREE Explaining Cluster Separation (Q3)
# Goal: figure out which stats separated the three clusters from K-Means
# Used a decision tree classifier
# Target was the cluster labels
# Max depth was 3 to keep the tree readable
# Focused on understanding splits, not predicting perfectly

X_tree = df[features]
y_tree = df["cluster"]

# Split data into train and test sets
# Test size 25 percent, random_state made results repeatable
X_train, X_test, y_train, y_test = train_test_split(
    X_tree, y_tree, test_size=0.25, random_state=42
)

# Fitted the decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Printed classification report for the test set
# Shows precision, recall, and f1-score for each cluster
print("Decision Tree Classification Report")
print(classification_report(
    y_test,
    dt.predict(X_test),
    target_names=["Low Impact", "High Volume", "High Efficiency"]
))

# Checked which features mattered most for splitting
# Formula: importance = contribution of each stat to splits
# Higher value means stat was more important
importances = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances")
print(importances.round(4))

# Plotted feature importances as a bar chart
plt.figure(figsize=(8, 4))
importances.plot(kind="bar", color="mediumpurple", edgecolor="white")
plt.title("Decision Tree Feature Importances")
plt.ylabel("Importance")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# Plotted the decision tree
# Top-to-bottom view shows which stats split the clusters
plt.figure(figsize=(18, 7))
plot_tree(
    dt,
    feature_names=features,
    class_names=["Low Impact", "High Volume", "High Efficiency"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree Rules Explaining Cluster Separation Q3")
plt.tight_layout()
plt.show()

# Q1 : Team Drafting Quality
# Goal: identified which teams most consistently drafted players
#       with long and impactful careers
# Derived features:
#   avg_win_shares  = mean of win_shares per team
#   avg_games       = mean of games played per team
#   avg_years       = mean of career length in years per team
#   total_drafted   = total number of players drafted by the team
# Showed which teams produced high-impact or long-career players


# Calculated team-level draft performance
team_stats = (
    df.groupby("team")
    .agg(
        avg_win_shares=("win_shares", "mean"),
        avg_games     =("games", "mean"),
        avg_years     =("years_played", "mean"),
        total_drafted =("player", "count")
    )
    .reset_index()
    .sort_values("avg_win_shares", ascending=False)
)

# Displayed top 15 teams by average win shares
print("Team Drafting Performance (top 15 by avg win shares):")
display(team_stats.head(15))

# Bar chart of top 12 teams by average win shares
# Formula: height = avg_win_shares
# Showed which teams drafted the most impactful players on average
top12_ws = team_stats.head(12)
plt.figure(figsize=(13, 5))
plt.bar(top12_ws["team"], top12_ws["avg_win_shares"],
        color="steelblue", edgecolor="white")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Drafting Team")
plt.ylabel("Avg Win Shares per Drafted Player")
plt.title("Q1 - Teams That Drafted Players With the Highest Career Impact")
plt.tight_layout()
plt.show()

# Bar chart of top 12 teams by average career length
# Formula: height = avg_years
# Showed which teams drafted players who played the longest careers
top12_years = team_stats.sort_values("avg_years", ascending=False).head(12)
plt.figure(figsize=(13, 5))
plt.bar(top12_years["team"], top12_years["avg_years"],
        color="seagreen", edgecolor="white")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Drafting Team")
plt.ylabel("Average Years Played")
plt.title("Q1 - Teams That Drafted Players With Long Careers")
plt.tight_layout()
plt.show()

# Bubble chart of career length vs win shares
# Formulas:
#   X-axis = avg_years
#   Y-axis = avg_win_shares
#   Bubble size = total_drafted * 15
# Showed the relationship between career length and impact
# Larger bubbles represented teams that drafted more players
plt.figure(figsize=(11, 7))
plt.scatter(
    team_stats["avg_years"],
    team_stats["avg_win_shares"],
    s=team_stats["total_drafted"] * 15,
    alpha=0.6, color="coral", edgecolors="white"
)
for _, row in team_stats.iterrows():
    plt.annotate(row["team"],
                 (row["avg_years"], row["avg_win_shares"]),
                 fontsize=7, ha="center", va="bottom")
plt.xlabel("Average Career Length (Years)")
plt.ylabel("Average Win Shares")
plt.title("Q1 - Team Draft Quality: Longevity vs Performance")
plt.tight_layout()
plt.show()

# Q2 : OVERPERFORMERS AND UNDERPERFORMERS
# Goal: identify players who significantly exceeded or fell short
#       of expectations based on their draft position
# Technique: Anomaly Detection using Local Outlier Factor (LOF)
# Formula: LOF score measures density relative to nearest neighbors
#          lof_label = -1 if unusual, 1 if normal

# Created pick groups by draft range for comparison
# Formula: pick_group = pd.cut(overall_pick, bins=[0,5,10,...], labels=[...])
df["pick_group"] = pd.cut(
    df["overall_pick"],
    bins=[0, 5, 10, 15, 20, 25, 30, 999],
    labels=["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31+"]
)

# Calculated median win shares per draft pick to define over and under performers
# Formula: median_ws_by_pick = median(win_shares) for each overall_pick
median_ws_by_pick = df.groupby("overall_pick")["win_shares"].median()

# Ran LOF anomaly detection using draft pick and win shares
# Formula: lof_label = -1 if unusual, 1 if normal; lof_score = -negative_outlier_factor_
lof_features = df[["overall_pick", "win_shares"]].dropna()
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
df_lof = df.dropna(subset=["overall_pick", "win_shares"]).copy()
df_lof["lof_label"] = lof.fit_predict(df_lof[["overall_pick", "win_shares"]])
df_lof["lof_score"] = -lof.negative_outlier_factor_

# Classified anomalies as overperformers or underperformers relative to draft pick median
# Formula: direction = Overperformer if win_shares > median_ws_by_pick else Underperformer
df_lof["direction"] = df_lof.apply(
    lambda row: "Overperformer" if row["win_shares"] > median_ws_by_pick[row["overall_pick"]]
                else "Underperformer",
    axis=1
)
anomalies = df_lof[df_lof["lof_label"] == -1].copy()

# Displayed number of LOF anomalies and top overperformers
print(f"LOF anomalies detected: {len(anomalies)}")
print("\nTop Overperformers:")
display(anomalies[anomalies["direction"] == "Overperformer"]
        .sort_values("win_shares", ascending=False)
        [["player", "overall_pick", "pick_group", "win_shares", "lof_score"]]
        .head(10))

# Displayed top underperformers
print("\nTop Underperformers:")
display(anomalies[anomalies["direction"] == "Underperformer"]
        .sort_values("win_shares")
        [["player", "overall_pick", "pick_group", "win_shares", "lof_score"]]
        .head(10))

# Plotted draft pick vs career win shares highlighting LOF anomalies
# X-axis = overall_pick, Y-axis = win_shares
# Color = anomaly type (green = overperformer, red = underperformer, gray = normal)
plt.figure(figsize=(12, 6))
normal_pts = df_lof[df_lof["lof_label"] == 1]
over_pts   = anomalies[anomalies["direction"] == "Overperformer"]
under_pts  = anomalies[anomalies["direction"] == "Underperformer"]

plt.scatter(normal_pts["overall_pick"], normal_pts["win_shares"],
            color="gray", alpha=0.4, s=20, label="Normal")
plt.scatter(over_pts["overall_pick"], over_pts["win_shares"],
            color="green", alpha=0.85, s=50, label="Overperformer (LOF)")
plt.scatter(under_pts["overall_pick"], under_pts["win_shares"],
            color="red", alpha=0.85, s=50, label="Underperformer (LOF)")

# Annotated top 5 overperformers
for _, row in over_pts.head(5).iterrows():
    plt.annotate(row["player"], (row["overall_pick"], row["win_shares"]),
                 fontsize=7, ha="left", va="bottom",
                 xytext=(4, 4), textcoords="offset points")

# Added median win shares line for reference
plt.axhline(df["win_shares"].median(), color="black", linestyle="--",
            linewidth=0.8, label="Overall median WS")
plt.xlabel("Draft Pick Number")
plt.ylabel("Career Win Shares")
plt.title("Q2: LOF Anomaly Detection: Over and Underperformers by Draft Pick")
plt.legend()
plt.tight_layout()
plt.show()

# Plotted win shares distribution by pick group
# X-axis = pick_group, Y-axis = win_shares
# Showed range and spread of win shares for each draft range
plt.figure(figsize=(12, 5))
df.boxplot(column="win_shares", by="pick_group", patch_artist=True, grid=False)
plt.suptitle("")
plt.title("Win Shares Distribution by Draft Pick Group")
plt.xlabel("Pick Group")
plt.ylabel("Win Shares")
plt.tight_layout()
plt.show()
