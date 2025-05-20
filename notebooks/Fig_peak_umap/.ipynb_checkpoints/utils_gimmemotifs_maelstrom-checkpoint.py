# Refactored code from gimmemotifs maelstrom
# last updated: 5/6/2025
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu, norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, scale
from statsmodels.stats.multitest import multipletests
from functools import partial
from multiprocessing import Pool

# Class to handle different motif activity prediction methods
class MotifActivityPredictor:
    """Base class for motif activity prediction."""
    
    @staticmethod
    def create(method_name, **kwargs):
        """Factory method to create activity predictors."""
        methods = {
            "hypergeom": HypergeomPredictor,
            "mwu": MWUPredictor,
            "rf": RFPredictor,
        }
        
        if method_name.lower() not in methods:
            raise ValueError(f"Unknown method: {method_name}. Available methods: {list(methods.keys())}")
        
        return methods[method_name.lower()](**kwargs)
    
    @staticmethod
    def list_predictors():
        """List available predictors."""
        return ["hypergeom", "mwu", "rf"]


class MWUPredictor(MotifActivityPredictor):
    """Mann-Whitney U test for motif activity prediction."""
    
    def __init__(self, **kwargs):
        self.act_ = None
        self.pref_table = "score"
        self.ptype = "classification"
        
    def fit(self, df_X, df_y):
        """
        Fit the predictor.
        
        Parameters
        ----------
        df_X : pandas.DataFrame
            Motif scores per region (peaks-by-motifs)
        df_y : pandas.DataFrame
            Cluster labels for each region
        
        Returns
        -------
        self
        """
        if not df_y.shape[0] == df_X.shape[0]:
            raise ValueError("Number of regions is not equal")
        if df_y.shape[1] != 1:
            raise ValueError("y needs to have 1 label column")

        # Calculate Mann-Whitney U p-values
        pvals = []
        clusters = df_y[df_y.columns[0]].unique()
        for cluster in clusters:
            pos = df_X[df_y.iloc[:, 0] == cluster]
            neg = df_X[df_y.iloc[:, 0] != cluster]
            p = []
            for m in df_X.columns:
                try:
                    p.append(mannwhitneyu(pos[m], neg[m], alternative="greater")[1])
                except Exception as e:
                    print(f"Motif {m} failed: {e}, setting to p = 1")
                    p.append(1)
            pvals.append(p)

        # Correct for multiple testing
        pvals = np.array(pvals)
        fpr = multipletests(pvals.flatten(), method="fdr_bh")[1].reshape(pvals.shape)

        # Create output DataFrame
        self.act_ = pd.DataFrame(-np.log10(fpr.T), columns=clusters, index=df_X.columns)
        
        return self


class HypergeomPredictor(MotifActivityPredictor):
    """Hypergeometric test for motif activity prediction."""
    
    def __init__(self, **kwargs):
        self.act_ = None
        self.pref_table = "count"
        self.ptype = "classification"
        
    def fit(self, df_X, df_y):
        """
        Fit the predictor using hypergeometric test.
        
        Parameters
        ----------
        df_X : pandas.DataFrame
            Motif counts per region (peaks-by-motifs)
        df_y : pandas.DataFrame
            Cluster labels for each region
        
        Returns
        -------
        self
        """
        if not df_y.shape[0] == df_X.shape[0]:
            raise ValueError("Number of regions is not equal")
        if df_y.shape[1] != 1:
            raise ValueError("y needs to have 1 label column")

        # Check if motif table contains integer counts
        if set(df_X.dtypes) != {np.dtype(int)}:
            # Convert to binary counts (presence/absence)
            df_X = (df_X > 0).astype(int)
            print("Warning: Converting motif scores to binary presence/absence")

        # Calculate hypergeometric p-values
        pvals = []
        clusters = df_y[df_y.columns[0]].unique()
        M = df_X.shape[0]  # Total number of regions
        
        for cluster in clusters:
            pos = df_X[df_y.iloc[:, 0] == cluster]  # Regions in this cluster
            neg = df_X[df_y.iloc[:, 0] != cluster]  # Regions not in this cluster

            pos_true = (pos > 0).sum(0)  # Number of regions in cluster with motif
            pos_false = (pos == 0).sum(0)  # Number of regions in cluster without motif
            neg_true = (neg > 0).sum(0)  # Number of regions not in cluster with motif

            p = []
            for pt, pf, nt in zip(pos_true, pos_false, neg_true):
                n = pt + nt  # Total regions with motif
                N = pt + pf  # Total regions in cluster
                x = pt - 1   # Successes - 1 for sf calculation
                p.append(hypergeom.sf(x, M, n, N))

            pvals.append(p)

        # Correct for multiple testing
        pvals = np.array(pvals)
        fpr = multipletests(pvals.flatten(), method="fdr_bh")[1].reshape(pvals.shape)

        # Create output DataFrame
        self.act_ = pd.DataFrame(-np.log10(fpr.T), columns=clusters, index=df_X.columns)
        
        return self


class RFPredictor(MotifActivityPredictor):
    """Random Forest for motif activity prediction."""
    
    def __init__(self, n_jobs=None, random_state=None, **kwargs):
        self.act_ = None
        self.pref_table = "score"
        self.ptype = "classification"
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def fit(self, df_X, df_y):
        """
        Fit the predictor using Random Forest.
        
        Parameters
        ----------
        df_X : pandas.DataFrame
            Motif scores per region (peaks-by-motifs)
        df_y : pandas.DataFrame
            Cluster labels for each region
        
        Returns
        -------
        self
        """
        if not df_y.shape[0] == df_X.shape[0]:
            raise ValueError("Number of regions is not equal")
        if df_y.shape[1] != 1:
            raise ValueError("y needs to have 1 label column")

        le = LabelEncoder()
        y = le.fit_transform(df_y.iloc[:, 0].values)

        clf = RandomForestClassifier(
            n_estimators=100, 
            n_jobs=self.n_jobs, 
            random_state=self.random_state
        )

        # Multiclass
        if len(le.classes_) > 2:
            orc = OneVsRestClassifier(clf)
            orc.fit(df_X.values, y)
            importances = np.array([c.feature_importances_ for c in orc.estimators_]).T
        else:  # Only two classes
            clf.fit(df_X.values, y)
            importances = np.array([clf.feature_importances_, clf.feature_importances_]).T

        # Adjust sign based on difference in motif score distribution
        for i, _ in enumerate(le.classes_):
            diff = df_X.loc[y == i].quantile(q=0.75) - df_X.loc[y != i].quantile(q=0.75)
            sign = (diff >= 0) * 2 - 1  # Convert to +1/-1
            importances[:, i] *= sign

        # Create output DataFrame
        self.act_ = pd.DataFrame(
            importances,
            columns=le.inverse_transform(range(len(le.classes_))),
            index=df_X.columns,
        )
        
        return self


def _rank_int(series, c=3.0/8, stochastic=True):
    """
    Rank-based inverse normal transformation on pandas series.
    
    Parameters
    ----------
    series : pandas.Series
        Series of values to transform
    c : float, optional
        Constant parameter (Blom's constant)
    stochastic : bool, optional
        Whether to randomize rank of ties
        
    Returns
    -------
    pandas.Series
    """
    # Check input
    assert isinstance(series, pd.Series)
    
    # Set seed
    np.random.seed(123)
    
    # Take original series indexes
    orig_idx = series.index
    
    # Drop NaNs
    series = series.loc[~pd.isnull(series)]
    
    # Get ranks
    if stochastic:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series
        rank = rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = rankdata(series, method="average")
    
    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)
    
    # Convert rank to normal distribution
    transformed = rank.apply(_rank_to_normal, c=c, n=len(rank))
    
    return transformed[orig_idx]


def _rank_to_normal(rank, c, n):
    """
    Convert rank to normal distribution.
    
    Parameters
    ----------
    rank : float
        Rank value
    c : float
        Constant parameter
    n : int
        Total number of items
        
    Returns
    -------
    float
    """
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return norm.ppf(x)


def _rankagg_int(df):
    """
    Rank aggregation using inverse normal transform and Stouffer's method.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with values to be ranked and aggregated
        
    Returns
    -------
    pandas.DataFrame
    """
    # Convert values to ranks
    df_int = df.apply(_rank_int)
    # Combine z-score using Stouffer's method
    df_int = (df_int.sum(1) / np.sqrt(df_int.shape[1])).to_frame()
    df_int.columns = ["z-score"]
    return df_int


def df_rank_aggregation(df, dfs, method="int_stouffer", ncpus=1):
    """
    Perform rank aggregation across multiple predictors.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Original input data
    dfs : dict
        Dictionary of DataFrames from different predictors
    method : str, optional
        Aggregation method, currently only "int_stouffer" is implemented
    ncpus : int, optional
        Number of CPU cores to use
        
    Returns
    -------
    pandas.DataFrame
    """
    df_p = pd.DataFrame(index=list(dfs.values())[0].index)
    names = list(dfs.values())[0].columns
    
    # Prepare data for aggregation
    dfs_list = [
        pd.concat([v[col].rename(k) for k, v in dfs.items()], axis=1)
        for col in names
    ]
    
    # Process with multiprocessing if needed
    if ncpus > 1 and len(dfs_list) > 1:
        with Pool(processes=ncpus) as pool:
            ret = pool.map(_rankagg_int, dfs_list)
    else:
        ret = [_rankagg_int(df_item) for df_item in dfs_list]
    
    # Combine results
    for name, result in zip(names, ret):
        df_p[name] = result["z-score"]
    
    # Rename columns to indicate they are z-scores
    df_p.columns = ["z-score " + c for c in df_p.columns]
    
    return df_p


def compute_motif_enrichment(
    peaks_motifs_matrix,
    cluster_labels,
    methods=None,
    ncpus=1,
    random_state=None
):
    """
    Compute motif enrichment scores (clusters-by-motifs) from peaks-by-motifs matrix.
    
    Parameters
    ----------
    peaks_motifs_matrix : pandas.DataFrame
        Matrix with motif scores for each peak
    cluster_labels : pandas.DataFrame or pandas.Series
        Cluster labels for each peak
    methods : list, optional
        List of activity prediction methods to use
    ncpus : int, optional
        Number of CPU cores to use
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    pandas.DataFrame
        Clusters-by-motifs matrix with enrichment scores
    """
    if isinstance(cluster_labels, pd.Series):
        cluster_labels = pd.DataFrame(cluster_labels)
    
    # Ensure that indices match
    if not all(peaks_motifs_matrix.index == cluster_labels.index):
        raise ValueError("Peak indices in motif matrix and cluster labels must match")
    
    # Default methods if none specified
    if methods is None:
        methods = ["hypergeom", "mwu", "rf"]
    
    # Run each method and collect results
    result_dfs = {}
    for method in methods:
        print(f"Running method: {method}")
        predictor = MotifActivityPredictor.create(
            method, 
            n_jobs=ncpus, 
            random_state=random_state
        )
        
        # Choose appropriate matrix type (scores or binary counts)
        if predictor.pref_table == "count" and predictor.ptype == "classification":
            # For methods that need counts, convert scores to binary presence/absence
            X = (peaks_motifs_matrix > 0).astype(int)
        else:
            X = peaks_motifs_matrix
            
        # Fit the predictor
        predictor.fit(X, cluster_labels)
        
        # Store the result
        result_dfs[method] = predictor.act_
    
    # If only one method was used, return its result directly
    if len(methods) == 1:
        return result_dfs[methods[0]]
    
    # Aggregate results from multiple methods
    final_scores = df_rank_aggregation(
        cluster_labels, 
        result_dfs, 
        method="int_stouffer", 
        ncpus=ncpus
    )
    
    # Add percentage of peaks with motif per cluster
    motif_presence = (peaks_motifs_matrix > 0).astype(int)
    freq_by_cluster = motif_presence.join(cluster_labels).groupby(cluster_labels.columns[0]).mean() * 100
    freq_by_cluster = freq_by_cluster.T
    freq_by_cluster = freq_by_cluster.rename(columns={col: f"{col} % with motif" for col in freq_by_cluster.columns})
    
    # Combine with enrichment scores
    final_result = pd.concat([final_scores, freq_by_cluster], axis=1)
    
    return final_result


# Example usage:
# 1. Load your peaks-by-motifs matrix
# peaks_motifs = pd.read_csv("motif.score.txt.gz", index_col=0, sep="\t")
# 
# 2. Load your cluster labels
# clusters = pd.read_csv("cluster_labels.txt", index_col=0, sep="\t")
# 
# 3. Compute the enrichment scores
# enrichment_scores = compute_motif_enrichment(
#     peaks_motifs, 
#     clusters, 
#     methods=["hypergeom", "mwu", "rf"], 
#     ncpus=4
# )
# 
# 4. Save the results
# enrichment_scores.to_csv("motif_enrichment_scores.txt", sep="\t")