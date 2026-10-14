
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors
import time



# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "Datasets"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

GDP_FILE = PROJECT_DIR / "gdp_growth_country_year_updated(3).csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


# ==========================================
# 2. LOAD GDP DATASET
# ==========================================

def load_gdp_data(file_path):
    """Load and validate the processed GDP dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"GDP dataset not found: {file_path}\n"
            "Check the filename and folder location."
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "Country Name",
        "Country Code",
        "Year",
        "GDP_Growth"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["GDP_Growth"] = pd.to_numeric(
        df["GDP_Growth"], errors="coerce"
    )

    print("\nGDP dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


# ==========================================
# 3. DATASET OVERVIEW
# ==========================================

def dataset_overview(df):
    """Display basic information about the dataset."""

    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print(
        "\nUnique country codes:",
        df["Country Code"].nunique()
    )

    print(
        "\nYear range:",
        df["Year"].min(),
        "to",
        df["Year"].max()
    )


# ==========================================
# 4. DATA QUALITY ANALYSIS
# ==========================================

def analyze_data_quality(df):
    """Save missing-value and duplicate reports."""

    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isnull().sum().values,
        "Missing_Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    missing_path = RESULTS_DIR / "missing_values_report.csv"
    missing_report.to_csv(missing_path, index=False)

    # Check duplicate country-year combinations.
    duplicates = df.duplicated(
        subset=["Country Code", "Year"],
        keep=False
    )

    duplicate_records = df.loc[duplicates]

    duplicate_path = RESULTS_DIR / "duplicate_country_years.csv"
    duplicate_records.to_csv(duplicate_path, index=False)

    print("\nMissing-value report saved:", missing_path)
    print("Duplicate country-year records:", len(duplicate_records))

    return missing_report


# ==========================================
# 5. DESCRIPTIVE STATISTICS
# ==========================================

def descriptive_analysis(df):
    """Calculate GDP growth statistics."""

    valid_gdp = df.dropna(subset=["GDP_Growth"])

    print("\n" + "=" * 50)
    print("GDP GROWTH STATISTICS")
    print("=" * 50)

    print(valid_gdp["GDP_Growth"].describe())

    summary = (
        valid_gdp.groupby("Country Code")["GDP_Growth"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    summary = summary.merge(
        valid_gdp[
            ["Country Code", "Country Name"]
        ].drop_duplicates(subset=["Country Code"]),
        on="Country Code",
        how="left",
        validate="one_to_one"
    )

    summary_path = RESULTS_DIR / "gdp_country_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nCountry summary saved:", summary_path)


# ==========================================
# 6. GDP GROWTH VISUALIZATION
# ==========================================

def plot_gdp_distribution(df):
    """Plot the distribution of available GDP growth values."""

    values = df["GDP_Growth"].dropna()

    if values.empty:
        print("No valid GDP growth values available for plotting.")
        return

    plt.figure(figsize=(9, 5))

    sns.histplot(values, bins=40, kde=True)

    plt.title("Distribution of GDP Growth Across Country-Years")
    plt.xlabel("GDP Growth (Annual %)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plot_path = RESULTS_DIR / "gdp_growth_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    print("GDP distribution plot saved:", plot_path)



# ==========================================
# 7. CLUSTERING ANALYSIS
# ==========================================

def prepare_clustering_data(df):
    """
    Prepare country-level GDP features for clustering.

    Each row represents one country.
    Features summarize GDP growth across available years.
    """

    valid_df = df.dropna(
        subset=["Country Code", "Year", "GDP_Growth"]
    ).copy()

    # Remove duplicate country-year observations.
    valid_df = valid_df.drop_duplicates(
        subset=["Country Code", "Year"]
    )

    # Create one row per country using GDP growth statistics.
    country_features = (
        valid_df.groupby(["Country Code", "Country Name"])
        .agg(
            GDP_Mean=("GDP_Growth", "mean"),
            GDP_Median=("GDP_Growth", "median"),
            GDP_Std=("GDP_Growth", "std"),
            GDP_Min=("GDP_Growth", "min"),
            GDP_Max=("GDP_Growth", "max"),
            GDP_Years=("GDP_Growth", "count")
        )
        .reset_index()
    )

    # Remove countries with insufficient observations.
    country_features = country_features[
        (country_features["GDP_Years"] >= 5)
    ].copy()

    feature_columns = [
        "GDP_Mean",
        "GDP_Median",
        "GDP_Std",
        "GDP_Min",
        "GDP_Max",
        "GDP_Years"
    ]

    X = country_features[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove features that contain no valid values.
    X = X.dropna(axis=1, how="all")
    feature_columns = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("\nCountry-level clustering dataset:", X_scaled.shape)

    return country_features, feature_columns, X_scaled


# ==========================================
# 8. PCA VISUALIZATION
# ==========================================

def reduce_dimensions(X_scaled):
    """Reduce standardized features to two PCA components."""

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_.sum()

    print(
        "\nPCA explained variance:",
        round(explained_variance * 100, 2),
        "%"
    )

    return X_pca


# ==========================================
# 9. CLUSTERING ALGORITHMS
# ==========================================

def run_clustering(country_features, X_scaled, X_pca):
    """Fit clustering algorithms and evaluate their results."""

    n_samples = len(X_scaled)

    if n_samples < 3:
        raise ValueError(
            "At least 3 countries with valid features are required."
        )

    results = {}
    evaluation = []

    # K-Means
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    start = time.perf_counter()
    results["KMeans"] = kmeans.fit_predict(X_scaled)
    kmeans_time = time.perf_counter() - start

    # Agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=3)

    start = time.perf_counter()
    results["Agglomerative"] = agglomerative.fit_predict(X_scaled)
    agglomerative_time = time.perf_counter() - start

    # DBSCAN
    # These parameters are starting values, not guaranteed optimal.
    dbscan = DBSCAN(eps=1.5, min_samples=5)

    start = time.perf_counter()
    results["DBSCAN"] = dbscan.fit_predict(X_scaled)
    dbscan_time = time.perf_counter() - start

    # Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=3,
        random_state=42,
        n_init=5
    )

    start = time.perf_counter()
    results["GMM"] = gmm.fit_predict(X_scaled)
    gmm_time = time.perf_counter() - start

    runtimes = {
        "KMeans": kmeans_time,
        "Agglomerative": agglomerative_time,
        "DBSCAN": dbscan_time,
        "GMM": gmm_time
    }

    for algorithm, labels in results.items():

        unique_labels = set(labels)
        noise_count = (
            int(np.sum(labels == -1))
            if algorithm == "DBSCAN"
            else 0
        )

        # Exclude DBSCAN noise from the internal cluster metrics.
        mask = labels != -1 if algorithm == "DBSCAN" else np.ones(
            len(labels), dtype=bool
        )

        valid_labels = labels[mask]
        valid_X = X_scaled[mask]
        cluster_count = len(set(valid_labels))

        if 2 <= cluster_count < len(valid_X):
            silhouette = silhouette_score(valid_X, valid_labels)
            davies_bouldin = davies_bouldin_score(
                valid_X, valid_labels
            )
            calinski_harabasz = calinski_harabasz_score(
                valid_X, valid_labels
            )
        else:
            silhouette = np.nan
            davies_bouldin = np.nan
            calinski_harabasz = np.nan

        evaluation.append({
            "Algorithm": algorithm,
            "Clusters": cluster_count,
            "Noise_Points": noise_count,
            "Silhouette_Score": silhouette,
            "Davies_Bouldin_Score": davies_bouldin,
            "Calinski_Harabasz_Score": calinski_harabasz,
            "Runtime_Seconds": runtimes[algorithm]
        })

        country_features[f"{algorithm}_Cluster"] = labels

    evaluation_df = pd.DataFrame(evaluation)

    evaluation_path = RESULTS_DIR / "clustering_evaluation.csv"
    evaluation_df.to_csv(evaluation_path, index=False)

    country_features.to_csv(
        RESULTS_DIR / "country_clustering_results.csv",
        index=False
    )

    print("\nCLUSTERING EVALUATION")
    print(evaluation_df.round(4).to_string(index=False))

    print("\nEvaluation saved:", evaluation_path)
    print(
        "Country clusters saved:",
        RESULTS_DIR / "country_clustering_results.csv"
    )

    return results, evaluation_df


# ==========================================
# 10. CLUSTER VISUALIZATION
# ==========================================

def plot_clusters(X_pca, results):
    """Plot each algorithm's clusters using the two PCA components."""

    for algorithm, labels in results.items():

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            cmap="tab10",
            s=55,
            alpha=0.8
        )

        plt.title(f"{algorithm} Clustering (PCA Projection)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster Label")
        plt.tight_layout()

        plot_path = RESULTS_DIR / f"{algorithm.lower()}_clusters.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print("Cluster plot saved:", plot_path)


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():
    print("=" * 55)
    print("MICROFINANCE AND ECONOMIC GROWTH")
    print("CLUSTERING ANALYSIS")
    print("=" * 55)

    gdp = load_gdp_data(GDP_FILE)

    dataset_overview(gdp)
    analyze_data_quality(gdp)
    descriptive_analysis(gdp)
    plot_gdp_distribution(gdp)

    country_features, feature_columns, X_scaled = (
        prepare_clustering_data(gdp)
    )

    X_pca = reduce_dimensions(X_scaled)

    results, evaluation_df = run_clustering(
        country_features,
        X_scaled,
        X_pca
    )

    plot_clusters(X_pca, results)

    print("\nClustering analysis completed.")
    print("Check the results folder for all output files.")


if __name__ == "__main__":
    main()
