"""
This is the main function for the data clustering machine
Author: Yonatan Eisenberg
"""


import logging
from configparser import ConfigParser
import argparse
import os
import sys
import pandas as pd
import numpy as np
import recordlinkage as rl
from recordlinkage.preprocessing import clean
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# read config file
config_object = ConfigParser()
config_object.read("config.ini")


def init_compare(args, logger):
    """
    This function initiates the comparing algorithm and returns a compare object
    """
    compare = rl.Compare()

    for col in args.compare_cols:
        for al in args.algos:
            compare.string(col, col, al, label=f'{col}_{al}')

    logger.info('initiated the compare object')
    return compare


def index_df(df, args, logger):
    """
    This function gets the candidate pairs to be compared according to the input argument index_by
    """

    indexer = rl.Index()

    for method, col in args.index_by.items():
        if method in ['Block', 'SortedNeighbourhood']:
            eval(f'indexer.add(rl.index.{method}({col}))')
        else:
            eval(f'indexer.add(rl.index.{method}())')

    candidate_pairs = indexer.index(df)

    logger.info('Indexed the dataset')
    return candidate_pairs


def preprocess_raw_df(raw_df, args, logger):
    """
    This function gets a dataframe containing raw data and preprocess it
    """

    prcs_df = raw_df.copy()

    # refactor Nan values in the about feature
    prcs_df['about'] = prcs_df.about.apply(lambda x: np.nan if x == 'unavailable' else x)

    # lower case columns we'll compare on
    for col in args.compare_cols:
        prcs_df[col] = clean(prcs_df[col])

    # remove duplicates if requested
    if args.remove_duplicates:

        # sort by sort_cols
        asc = (False for _ in args.sort_cols)
        prcs_df = prcs_df.sort_values(args.sort_cols, ascending=asc)

        # remove duplicates by drop_cols
        prcs_df.drop_duplicates(subset=args.drop_cols, inplace=True)

    logger.info('Preprocessed the raw dataset')
    return prcs_df


def val_data(raw_df, args, logger):
    """
    This function validates that the columns names in the input data appears in the raw dataset.
    """

    # validate sort_cols, drop_cols and compare_cols
    for input_col in set(args.sort_cols + args.drop_cols + args.compare_cols):
        if input_col not in raw_df.columns:
            logger.debug('The columns used as input doesn\'t match the dataset')
            exit()

    # validate index_by
    for method, col in args.index_by.items():
        if method in ['Block', 'SortedNeighbourhood']:
            if col not in raw_df.columns:
                logger.debug('The columns used as input doesn\'t match the dataset')
                exit()

    logger.info('Validated input columns match the dataset')


def get_data(args, logger):
    """
    This function validate the path to the files and returns a pandas dataframe containing all the data. Quits the
    program if the folder or the expected csv files doesn't exists .
    """
    df_list = []

    # verify folder path exists
    folder_path = config_object['const']['data_path_prefix'] + args.city
    if os.path.exists(folder_path):

        # load the data
        for supplier in config_object["const"]["source_list"].split(','):
            filepath = folder_path + '/' + args.city.lower() + '_' + supplier + '.csv'
            if os.path.exists(filepath):
                t_df = pd.read_csv(filepath, encoding='UTF-8')
                t_df['data_source'] = supplier
                df_list.append(t_df)
    else:
        logger.debug('The folder doesn\'t exists, terminating program')
        print('Error, missing folder')
        exit()

    if len(df_list) == 0:
        logger.debug('No files were found in the folder, terminating program')
        print('Error, missing files')
        exit()

    # concat the dataframes
    raw_df = pd.concat(df_list).reset_index().drop(columns='index')

    logger.info('Loaded the raw data')

    return raw_df


def parse_args(args, logger):
    """
    This function initialize the parser and the input parameters
    """

    my_parser = argparse.ArgumentParser(description=config_object['params']['description'])
    my_parser.add_argument('--city', '-c',
                           required=True,
                           default=None,
                           type=str, nargs='+',
                           help=config_object['params']['city_help'])

    my_parser.add_argument('--remove_duplicates', '-rd',
                           default=False, action=argparse.BooleanOptionalAction,
                           type=bool,
                           help=config_object['params']['remove_duplicates_help']
                           )

    my_parser.add_argument('--sort_cols', '-sc',
                           default=config_object['params']['sort_cols'].split(),
                           type=str, nargs='+',
                           help=config_object['params']['sort_cols_help']
                           )

    my_parser.add_argument('--drop_cols', '-dc',
                           default=config_object['params']['drop_cols'].split(),
                           type=str, nargs='+',
                           help=config_object["params"]["drop_cols_help"])

    my_parser.add_argument('--index_by', '-ib',
                           default=config_object['params']['index_by'],
                           type=lambda v: {k: v for k, v in (x.split(':') for x in v.split(','))},
                           help=config_object["params"]["index_by_help"])

    my_parser.add_argument('--compare_cols', '-cc',
                           default=config_object['params']['compare_cols'].split(),
                           type=str, nargs='+',
                           help=config_object["params"]["compare_cols_help"])

    my_parser.add_argument('--algos', '-al',
                           default=config_object['params']['algos'].split(),
                           type=str, nargs='+',
                           help=config_object["params"]["algos_help"])

    logger.info('Parsed arguments')
    return my_parser.parse_args()


def init_logger():
    """
    This function initialize the logger and returns its handle
    """
    # todo: put format string in the config file
    log_formatter = logging.Formatter('%(levelname)s-%(asctime)s-FUNC:%(funcName)s-LINE:%(lineno)d-%(message)s')
    logger = logging.getLogger('log')
    logger.setLevel('DEBUG')
    file_handler = logging.FileHandler('log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


def main():

    # config logger
    logger = init_logger()
    logger.info('STARTED RUNNING')

    # parse args
    args = parse_args(sys.argv[1:], logger)
    args.city = ' '.join(args.city)
    logger.info(f'args={args}')

    # get the data
    raw_df = get_data(args, logger)

    # validate the data match input parameters
    val_data(raw_df, args, logger)

    # preprocess the data
    df = preprocess_raw_df(raw_df, args, logger)

    # get candidate pairs
    candidate_pairs = index_df(df, args, logger)

    # initialize the compare object
    compare = init_compare(args, logger)

    # compute the features
    features = compare.compute(candidate_pairs, df)
        # .rename(columns={'level_0': 'index1', 'level_1': 'index2'}, inplace=True)

    # save the df
    save_path = config_object['const']['data_path_prefix'] + args.city + '/extanded_features.csv'
    features.to_csv(save_path)
    logger.info('FINISHED RUNNING')


if __name__ == '__main__':
    main()

