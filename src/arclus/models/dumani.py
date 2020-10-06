"""Precomputed rankings by Dumani et al."""
from typing import Sequence

from arclus.models.base import RankingMethod
from arclus.utils import load_assignments_with_numeric_relevance


class Dumani(RankingMethod):
    """Precomputed ranking by Dumani et al."""

    def __init__(
        self,
        column: str,
    ):
        """
        Initialize the method.

        :param column:
            The ranking method. Must be in {"first512Tokens", "slidingWindow", "sentences"}.
        """
        self.sort_col = rf"P(\pi_j|q)_{column}"
        self.cluster_col = f"premiseClusterID_{column}"
        df = load_assignments_with_numeric_relevance()
        df["premise_text_length"] = df["premise_text"].fillna('').apply(len)
        self.data = df[["premise_id", "premise_text_length", self.sort_col, self.cluster_col]].copy()

    def rank(self, claim_id: int, premise_ids: Sequence[str], k: int) -> Sequence[str]:  # noqa: D102
        # select precompute data for this claim
        selection = self.data.loc[self.data["premise_id"].isin(premise_ids)]
        # select longest premise from each cluster
        result = []
        for _, cluster in selection.groupby(by=self.cluster_col):
            result.extend(cluster.sort_values(by="premise_text_length", ascending=False).head()["premise_id"].tolist())
        # sort premises by score
        return selection.loc[selection["premise_id"].isin(result)].sort_values(by=self.sort_col, ascending=False)["premise_id"].tolist()
