"""Test for evaluation utilities."""
import pprint

import numpy
import pandas

from arclus.evaluation import evaluate


def test_evaluate():
    """Test evaluate(...)."""
    num_claims, num_premises, num_pairs = 16, 33, 101
    gt_assignment = pandas.DataFrame(data=dict(
        claim_id=numpy.random.randint(num_claims, size=(num_pairs,)),
        premise_id=numpy.random.randint(num_premises, size=(num_pairs,)),
    ))

    assignment = numpy.random.uniform(size=(num_claims, num_premises))

    evaluation_result = evaluate(
        pairwise_scores=assignment,
        gt_assignment=gt_assignment,
    )
    assert isinstance(evaluation_result, dict)
    pprint.pprint(evaluation_result)
