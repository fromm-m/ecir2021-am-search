import numpy as np
import torch
from sklearn.cluster import KMeans

from arclus.similarity import LpSimilarity, get_most_similar


def clustering(args, claim_representation, premise_representations):
    n_clusters = round(len(premise_representations) / 1)
    n_clusters = max([n_clusters, args.k])
    # cluster all premises, n_clusters can be chosen
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(premise_representations)
    prepare_centers = torch.from_numpy(kmeans.cluster_centers_)
    # choose representative of each cluster
    if args.repr == "center":
        # choose nearest to cluster centers as representative for each cluster
        repr = [
            get_most_similar(torch.reshape(center, (1, len(center))), premise_representations.double(), 1,
                             LpSimilarity()) for center in prepare_centers]
    else:
        premises_per_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

        # choose representative, here: nearest to claim
        repr = [
            get_most_similar(claim_representation,
                             premise_representations[premises_per_cluster[i]], 1,
                             LpSimilarity()) for i in range(kmeans.n_clusters)]
    # format representatives
    representatives = torch.cat([x[0].reshape(-1, x[0].shape[-1]) for x in repr])
    repr_ind = torch.cat([x[1].reshape(-1) for x in repr])
    return repr_ind, representatives
