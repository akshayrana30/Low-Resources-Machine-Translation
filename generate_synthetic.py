import os
import argparse
import logging
from evaluator import generate_predictions


def main():
    parser = argparse.ArgumentParser(
        'script to create synthetic data for backtranslation.')
    parser.add_argument('--input', help='file to be translated')
    parser.add_argument('--output', help='path to outputs - will store files here',
                        required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Todo: Remember to modified the path of checkpoints in evaluator.py
    generate_predictions(args.output, args.output)


if __name__ == '__main__':
    main()
