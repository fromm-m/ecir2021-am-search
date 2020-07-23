"""Evaluation utilities."""
from typing import Any, Mapping, Union

import numpy
import pandas
import torch
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


def evaluate(
    pairwise_scores: Union[torch.FloatTensor, numpy.ndarray],
    gt_assignment: pandas.DataFrame,
    classification_threshold: float = 0.5,
) -> Mapping[str, Any]:
    """
    Evaluate predicted similarities between all claims and premises.

    :param pairwise_scores: shape: (num_claims, num_premises)
        Matrix of pairwise assignment scores.
    :param gt_assignment: columns: ['claim_id', 'premise_id']
        The dataframe of ground truth assignments.
    :param classification_threshold:
        A threshold for assignment decision based on scores.

    :return:
        A dictionary of results.
    """
    if torch.is_tensor(pairwise_scores):
        pairwise_scores = pairwise_scores.cpu().numpy()

    num_claims, num_premises = pairwise_scores.shape

    # create dense target
    true_claim_id, true_premise_id = gt_assignment.loc[:, ['claim_id', 'premise_id']].values.T
    full_assignment = numpy.zeros(shape=(num_claims, num_premises), dtype=numpy.bool)
    full_assignment[true_claim_id, true_premise_id] = True
    flat_full_assignment = full_assignment.flatten()

    # classification evaluation
    classification_result = dict()
    classification_pred = pairwise_scores > classification_threshold
    flat_classification_pred = classification_pred.flatten()
    classification_result['report'] = classification_report(
        y_true=flat_full_assignment,
        y_pred=flat_classification_pred,
        output_dict=True,
    )

    return dict(
        classification_result=classification_result,
        roc_auc_score=roc_auc_score(
            y_true=flat_full_assignment,
            y_score=pairwise_scores.flatten(),
        ),
        average_precision_score=average_precision_score(
            y_true=flat_full_assignment,
            y_score=pairwise_scores.flatten(),
        ),
    )
