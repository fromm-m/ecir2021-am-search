import argparse
import itertools
import json
import logging
import pathlib
from typing import Union

import pandas


class DatasetReader:
    """General class for reading datasets."""

    #: The root directory.
    root: pathlib.Path

    def __init__(
        self,
        root: Union[pathlib.Path, str],
        claim_id: int = 0,
        premise_id: int = 0,
    ):
        if not isinstance(root, pathlib.Path):
            root = pathlib.Path(root)
        self.root = root
        self.offset_claim_id = claim_id
        self.offset_premise_id = premise_id
        self._claims = []
        self._premises = []
        self._claim_premises = []
        self._read()

    def _add_claim(self, text: str) -> int:
        """
        Add a claim to the dataset.

        :param text:
            The claim text.
        :return:
            The claim ID.
        """
        claim_id = len(self._claims) + self.offset_claim_id
        self._claims.append(dict(
            claim_text=text,
            claim_id=claim_id,
            source=self.name,
        ))
        return claim_id

    def _add_premise(self, text: str) -> int:
        """
        Add a premise to the dataset.

        :param text:
            The premise text.
        :return:
            The premise ID.
        """
        premise_id = len(self._premises) + self.offset_premise_id
        self._premises.append(dict(
            claim_text=text,
            claim_id=premise_id,
            source=self.name,
        ))
        return premise_id

    def _add_claim_premise(self, premise_id: int, claim_id: int) -> None:
        """A a link between premise_id and claim_id."""
        self._claim_premises.append(dict(
            premise_id=premise_id,
            claim_id=claim_id,
        ))

    def __str__(self):
        return f'Dataset(name={self.name}, num_claims={len(self._claims)}, num_premises={len(self._premises)})'

    @property
    def max_claim_id(self) -> int:
        return self.offset_claim_id + len(self._claims)

    @property
    def max_premise_id(self) -> int:
        return self.offset_premise_id + len(self._premises)

    @property
    def premises(self) -> pandas.DataFrame:
        return pandas.DataFrame(self._premises)

    @property
    def claims(self) -> pandas.DataFrame:
        return pandas.DataFrame(self._claims)

    @property
    def claims_premises(self) -> pandas.DataFrame:
        return pandas.DataFrame(self._claim_premises)

    @property
    def name(self) -> str:
        """The name of the dataset."""
        raise NotImplementedError

    def _read(self):
        """Read the dataset."""
        raise NotImplementedError


class DebatePediaOrgReader(DatasetReader):
    """DebatePediaOrg dataset."""

    def _read(self):
        for index, file_path in enumerate(self.root.iterdir()):
            with file_path.open(mode='r', errors='ignore') as json_data:
                data = json.load(json_data)
                for claim_data in data:
                    if len(claim_data['pros']) + len(claim_data['cons']) == 0:
                        logging.warning('Skipping empty file')
                        continue
                    claim_id = self._add_claim(text=claim_data['claimText'])
                    for premise_data in itertools.chain(claim_data['pros'], claim_data['cons']):
                        premise_id = self._add_premise(text=premise_data['premiseText'])
                        self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


class DebateOrgReader(DatasetReader):
    """DebateOrg dataset."""

    def _read(self):
        with self.root.open(mode='r', errors='ignore') as json_data:
            data = json.load(json_data)
            for claim_data in data:
                if len(claim_data['pros']) + len(claim_data['cons']) == 0:
                    logging.warning('Skipping empty file')
                    continue
                claim_id = self._add_claim(text=claim_data['title'])
                for premise_data in itertools.chain(claim_data['pros'], claim_data['cons']):
                    premise_id = self._add_premise(text=premise_data['text'])
                    self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


class DebateWiseReader(DatasetReader):
    """DebateWise dataset."""

    @property
    def name(self) -> str:
        return 'debatewise'

    def _read(self):
        for index, file_path in enumerate(self.root.iterdir()):
            with file_path.open(mode='r', errors='ignore') as json_data:
                data = json.load(json_data)
                if len(data['ArgumentList']) == 0:
                    logging.warning('Skipping empty file')
                    continue
                claim_id = self._add_claim(text=data['MetaData']['Title'])
                for premise_data in data['ArgumentList']:
                    premise_id = self._add_premise(text=premise_data['Argument']['Premise'][0])
                    self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


class IDebateOrgReader(DatasetReader):
    """iDebateOrg dataset."""

    @property
    def name(self) -> str:
        return 'iDebateOrg'

    def _read(self):
        for index, file_path in enumerate(self.root.iterdir()):
            with file_path.open(mode='r', errors='ignore') as json_data:
                data = json.load(json_data)
                for claim_data in data:
                    if len(claim_data['pros']) + len(claim_data['cons']) == 0:
                        logging.warning('Skipping empty file')
                        continue
                    claim_id = self._add_claim(text=claim_data['title'])
                    for premise_data, premise_text_field_name in zip(
                        (claim_data['pros'], claim_data['cons']),
                        ('title point pro claim', 'title point con claim')
                    ):
                        premise_id = self._add_premise(text=premise_data[premise_text_field_name])
                        self._add_claim_premise(premise_id=premise_id, claim_id=claim_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input Directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output Directory")
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    c_id, p_id = 0, 0
    datasets = []
    for name, dataset_root, cls in [
        ('DebatePediaOrg', input_dir / "debatePediaOrg" / "json_files", DebatePediaOrgReader),
        ('DebateOrg', input_dir / "debateOrg" / "debates.json", DebateOrgReader),
        ('DebateWise', input_dir / "debatewise", DebateWiseReader),
        ('iDebateOrg', input_dir / "iDebateOrg" / "json_files", IDebateOrgReader)
    ]:
        logging.info(f'Read {name} from {dataset_root}.')
        reader = cls(
            root=dataset_root,
            claim_id=c_id,
            premise_id=p_id,
        )
        logging.info(f'Read dataset: {str(reader)}')
        c_id, p_id = reader.max_claim_id, reader.max_premise_id
        datasets.append(reader)

    premises_df = pandas.concat([dataset.premises for dataset in datasets])
    claims_df = pandas.concat([dataset.claims for dataset in datasets])
    assignments_df = pandas.concat([dataset.claims_premises for dataset in datasets])

    print("Claims total: ", len(claims_df))
    print("Premises total: ", len(premises_df))

    for df, name in [
        (premises_df, 'premises.csv'),
        (claims_df, 'claims.csv'),
        (assignments_df, 'assignments.csv'),
    ]:
        df.to_csv(output_dir / name)


if __name__ == '__main__':
    main()
