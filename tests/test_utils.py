"""Tests for utility methods."""
import itertools
import random
import string
from typing import List

import pandas

from arclus.utils import concat_premise_claims, truncate


def _generate_random_words(num_words, max_word_length: int = 10) -> List[str]:
    """Generate a list of random words."""
    return [
        "".join(random.choices(string.ascii_letters, k=random.randrange(1, max_word_length)))
        for _ in range(num_words)
    ]


def test_truncate():
    """Test truncate()."""
    for limit, n_words in itertools.product(range(10, 30, 5), range(10, 100, 10)):
        # generate list of random words
        words = _generate_random_words(num_words=n_words)

        # join to text
        text = " ".join(words)

        # truncate
        trunc_text = truncate(text, limit)

        # check output
        assert trunc_text == " ".join(words[:limit])


def test_concat_premise_claims():
    """Test concat_premise_claims()."""
    # Generate premises, long enough to get truncated
    num_premises = 7
    max_premise_words = 3
    premise_texts = [
        " ".join(_generate_random_words(num_words=max_premise_words + random.randrange(1, max_premise_words)))
        for _ in range(num_premises)
    ]
    premises_df = pandas.DataFrame(
        data=dict(
            premise_id=range(num_premises),
            premise_text=premise_texts,
        )
    )

    # Generate claims, long enough to get truncated
    num_claims = 3
    max_claim_words = 2
    claim_texts = [
        " ".join(_generate_random_words(num_words=max_claim_words + random.randrange(1, max_claim_words)))
        for _ in range(num_claims)
    ]
    claims_df = pandas.DataFrame(
        data=dict(
            claim_id=range(num_claims),
            claim_text=claim_texts,
        )
    )

    # Generate random assignment, many-to-many
    num_assignments = num_claims + num_claims
    assignment_df = pandas.DataFrame(
        data=dict(
            premise_id=random.choices(range(num_premises), k=num_assignments),
            claim_id=random.choices(range(num_claims), k=num_assignments),
        )
    )

    # call method
    texts = concat_premise_claims(
        premises_df=premises_df,
        claims_df=claims_df,
        assignment_df=assignment_df,
        max_premise_words=max_premise_words,
        max_claim_words=max_claim_words,
    )

    # check type
    assert isinstance(texts, pandas.Series)
    assert all(isinstance(text, str) for text in texts)

    # check concat symbol
    assert all(' ||| ' in text for text in texts)

    # check number of words
    for part, num in enumerate([max_premise_words, max_claim_words]):
        assert texts.apply(lambda s: len(s.split(' ||| ')[part].split()) <= num).all()
