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
    adf = []
    for file_path in files:
        df = pandas.read_csv(file_path, sep='\t')
        df['file_path'] = file_path
        adf.append(df)
    df = pandas.concat(
        objs=adf,
        ignore_index=True,
    ).fillna('')
    df['softmax'] = df['softmax'].apply(lambda x: True if x == 1.0 else (False if x == 0.0 else ''))
    mask = df['method'] == 'dumani'
    df.loc[mask, 'similarity'] = df.loc[mask, 'column']
    summary = df.groupby(
        by=[
            'method',
            'similarity',
            'cluster_ratio',
            'cluster_representative',
            'softmax',
            'k',
        ]).agg(dict(mnDCG='mean'))
    summary = summary.unstack()
    print(summary)
    print(summary.to_csv())


if __name__ == '__main__':
    main()
