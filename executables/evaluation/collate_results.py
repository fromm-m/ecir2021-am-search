"""Collect results from baseline evaluation and generate summary."""
import argparse
import pathlib

import pandas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_directory', default='./results', type=str)
    args = parser.parse_args()
    input_root = pathlib.Path(args.result_directory).expanduser().absolute()
    files = list(input_root.glob('*.tsv'))
    if len(files) == 0:
        print('no input files')
        quit(-1)
    df = pandas.concat(
        objs=[pandas.read_csv(file_path, sep='\t') for file_path in files],
        ignore_index=True,
    ).fillna('None')
    summary = df.groupby(
        by=[
            'method',
            'similarity',
            'cluster_ratio',
            'cluster_representative',
            'k',
        ]).agg(dict(mnDCG='mean'))
    summary = summary.unstack()
    print(summary)
    print(summary.to_csv())


if __name__ == '__main__':
    main()
