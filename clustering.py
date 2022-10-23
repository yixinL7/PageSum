from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import numpy as np
from compare_mt.rouge.rouge_scorer import RougeScorer

scorer = RougeScorer(['rouge1'], use_stemmer=True)

def compute_rouge(cand, ref):
    score = scorer.score("\n".join(ref), "\n".join(cand))
    return score["rouge1"].fmeasure

def get_labels(mat, n_clusters=5):
    x = SpectralClustering(n_clusters=n_clusters, affinity="precomputed").fit(mat + 1)
    return x.labels_

def get_labels_ag(mat, n_clusters=5):
    x = AgglomerativeClustering(n_clusters=5, affinity="precomputed", linkage="single").fit(1 - mat)
    return x.labels_

def oracle_clustering(article, abstract, mat, n_clusters=5):
    scores = [compute_rouge([x], abstract) for x in article]
    idx = np.argsort(scores)
    # find the best first n_clusters
    mat = mat[idx[:n_clusters]]  # only keep the top n_clusters [n_clusters, n_sents]
    mat[:, idx[:n_clusters]] = -100
    clusters = [[x] for x in idx[:n_clusters].tolist()]
    max_num = len(article) // n_clusters + 1
    for _ in range(len(article) - n_clusters):
        nearest_per_cluster = mat.max(axis=0)
        ids = mat.argmax(axis=0)
        selected_id = np.argmax(nearest_per_cluster)
        cluster_id = ids[selected_id].item()
        clusters[cluster_id].append(selected_id.item())
        if len(clusters[cluster_id]) == max_num:
            # remove the full cluster
            mat[cluster_id, :] = -100
        mat[:, selected_id] = -100 # remove the selected id
    labels = [0] * len(article)
    for i, cluster in enumerate(clusters):
        for x in cluster:
            labels[x] = i
    return labels