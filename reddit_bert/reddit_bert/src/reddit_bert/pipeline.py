import argparse

from reddit_bert.commands import split_data, train, predict


def main(selected_stages):
    if 'split_data' in selected_stages:
        split_data()
    if 'train' in selected_stages:
        train()
    if 'predict' in selected_stages:
        predict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run pipeline stages.")
    parser.add_argument(
        'stages',
        nargs='+',
        help='Specify stages to run: split_data, train, predict',
    )
    args = parser.parse_args()
    main(args.stages)
