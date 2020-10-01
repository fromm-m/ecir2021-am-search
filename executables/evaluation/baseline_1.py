import argparse
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from arclus.evaluation import best_ranking, ndcg_score, split_clusters
from arclus.get_similar import LpSimilarity, Similarity, get_most_similar
from arclus.settings import CLAIMS_TEST_FEATURES, PREMISES_TEST_FEATURES, PREP_PREMISES_TEST
from arclus.utils import load_assignments_with_numeric_relevance


class RankingMethod:
    """Base class for ranking methods."""

    @abstractmethod
    def rank(
        self,
        queries: List[int],
        k: int,
    ) -> np.ndarray:
        """
        Return a sorted list of k premise IDs for each query claim.

        :param queries:
            The query claim IDs.
        :param k: >0
            The number of premises to return for each claim.

        :return: shape: (num_queries, k)
            The sorted list of premise IDs for each query claim.
        """
        raise NotImplementedError


class ZeroShotRanking(RankingMethod):
    """Abstract base class for zero-shot methods."""

    # Pre-computed representations
    claims: torch.FloatTensor
    premises: torch.FloatTensor

    def __init__(self, similarity: Similarity):
        """
        Initialize the method.

        :param similarity:
            The similarity to use for the representations.
        """
        self.similarity = similarity
        # Load pre-computed representations
        self.claims = torch.load(CLAIMS_TEST_FEATURES)
        self.premises = torch.load(PREMISES_TEST_FEATURES)


class ZeroShotKNN(ZeroShotRanking):
    """Rank according to similarity of pre-trained BERT representations."""

    def rank(self, queries: List[int], k: int) -> np.ndarray:  # noqa: D102
        num_queries = len(queries)
        result = np.empty(shape=(num_queries, k), dtype=np.int64)
        for i, claim_id in enumerate(queries):
            # get the claim representation
            claim_repr = self.claims[claim_id]
            # TODO: Evaluate only a subset of premises?
            result[i, :] = self.similarity.sim(
                claims=claim_repr,
                premises=self.premises,
            ).topk(k=k, largest=True, sorted=True).indices.numpy()
        return result


class ZeroShotClusterKNN(ZeroShotRanking):
    """Rank according to similarity of pre-trained BERT representations, return at most one premise for each cluster."""

    def __init__(
        self,
        similarity: Similarity,
        reduction_factor: float = 0.5,
        cluster_representative: str = 'closest-to-center',
    ):
        """
        Initialize the method.

        :param similarity:
            The similarity to use for the representations.
        :param reduction_factor:
            The relative number of premise clusters.
        """
        super().__init__(similarity=similarity)
        self.reduction_factor = reduction_factor
        self.cluster_representative = cluster_representative

    def _get_representatives(
        self,
        premises: torch.FloatTensor,
        assignment: torch.LongTensor,
        centroids: torch.FloatTensor,
        n_clusters: int,
        claim_repr: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Select cluster representatives for each cluster.

        :param premises: shape: (num_premises, dim)
            The premise representations.
        :param assignment: shape: (num_premises,)
            The cluster assignment
        :param centroids: shape: (num_premises,)
            The cluster centroids.
        :param n_clusters: >0
            The number of clusters.
        :param claim_repr: shape: (dim,)
            The claim representation.

        :return:
            A tuple repr, repr_id of the representations of the representative premises for each cluster and their
            corresponding premise IDs.
        """
        premise_ids = torch.arange(premises.shape[0])
        cluster_repr_id = premises.new_empty(n_clusters, dtype=torch.long)
        cluster_repr = premises.new_empty(n_clusters, premises.shape[-1])
        for i in range(n_clusters):
            if self.cluster_representative == 'closest-to-center':
                anchor = centroids[i]
            elif self.cluster_representative == 'closest-to-claim':
                anchor = claim_repr
            else:
                raise NotImplementedError
            mask = assignment == i
            premises_in_cluster = premises[mask]
            idx = self.similarity.sim(claims=anchor.unsqueeze(dim=0), premises=premises_in_cluster).argmax(dim=1)[0]
            cluster_repr[i, :] = premises_in_cluster[idx]
            cluster_repr_id[i] = premise_ids[mask][idx]
        return cluster_repr, cluster_repr_id

    def rank(self, queries: List[int], k: int) -> np.ndarray:  # noqa: D102
        # determine absolute number of clusters
        n_clusters = (self.reduction_factor * self.premises.shape[0])
        n_clusters = max([n_clusters, k])

        # Determine premise clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        assignment = torch.from_numpy(kmeans.fit_predict(self.premises.numpy()))
        centroids = torch.from_numpy(kmeans.cluster_centers_)

        num_queries = len(queries)
        result = np.empty(shape=(num_queries, k), dtype=np.int64)
        for i, claim_id in enumerate(queries):
            # get the claim representation
            claim_repr = self.claims[claim_id]

            # get cluster representations
            cluster_repr, cluster_repr_id = self._get_representatives(
                premises=self.premises,
                assignment=assignment,
                centroids=centroids,
                n_clusters=n_clusters,
                claim_repr=claim_repr,
            )

            # get closest cluster representatives
            top_clusters = self.similarity.sim(
                claims=claim_repr,
                premises=cluster_repr,
            ).topk(k=k, largest=True, sorted=True).indices.numpy()

            # choose representative of cluster
            result[i, :] = cluster_repr_id[top_clusters].numpy()
        return result


def main():
    """
    Calculates normalized discounted cumulative gain (nDCG) for all queries.
    """
    parser = argparse.ArgumentParser(description='Pre-compute BERT features.')
    parser.add_argument('--k', type=int, default=5, choices=[5, 10, ],
                        help='The first k elements in the ranking should be considered')
    parser.add_argument('--pad', type=bool, default=True,
                        help='Should the ranking be padded with 0s until k positions are reached')
    args = parser.parse_args()
    k = args.k
    df_premises = pd.read_csv(PREP_PREMISES_TEST)
    df = load_assignments_with_numeric_relevance()
    claims_representations = torch.load(CLAIMS_TEST_FEATURES)
    premises_representations = torch.load(PREMISES_TEST_FEATURES)

    ndcg_list = []
    # iterate over all claims
    for id in df["claim_id"].unique():
        # locate all premises which are assigned to the current claim
        premises = df.loc[df["claim_id"] == id]
        premises_ids = premises["premise_id"].to_numpy(dtype=str)

        # get the premises representations
        premise_representation_list = [premises_representations[x] for x in premises_ids]
        assert len(premise_representation_list) == len(premises_ids)
        premises_repr = torch.stack(premise_representation_list)
        # get the claim representation
        claim_repr = claims_representations[id].unsqueeze(dim=0)

        # find knn premises given a claim
        k_premises, k_indices = get_most_similar(
            claims=claim_repr,
            premises=premises_repr,
            k=k,
            similarity=LpSimilarity()
        )

        # generate the ranking (relevance) of the knn premises
        predicted_ranking = df.iloc[k_indices.squeeze().numpy()].relevance.to_numpy(dtype=int)

        # groundtruth
        ordered_gt_cluster_ids = premises["premiseClusterID_groundTruth"].sort_values().dropna().unique()
        splitted_gt_clusters = split_clusters(premises, ordered_gt_cluster_ids, "premiseClusterID_groundTruth")
        gt_ranking = best_ranking(splitted_gt_clusters)
        gt_ranking.sort(reverse=True)

        # calculate nDCG for the given claim
        ndcg_list.append(ndcg_score(y_score=predicted_ranking, y_true=gt_ranking, k=k, pad=args.pad))

    print("task _b;", "algorithm:", "baseline_1", "nDCG@", k, np.array(ndcg_list).mean())


if __name__ == '__main__':
    main()
