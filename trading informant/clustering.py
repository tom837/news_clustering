from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS


def agglo_clustering(dist, n_clusters):
    cluster = AgglomerativeClustering(n_clusters=n_clusters,affinity='precomputed',linkage='complete')
    cluster.fit(dist)
    return cluster


def dbscan_clustering(dist, eps, min_samples):
    cluster = DBSCAN(eps = eps, min_samples = min_samples, metric = 'precomputed')
    cluster.fit(dist)
    return cluster

def optics_clustering(dist,min_samples):
    cluster = OPTICS(min_samples=min_samples, metric="precomputed")
    cluster.fit(dist)
    return cluster

