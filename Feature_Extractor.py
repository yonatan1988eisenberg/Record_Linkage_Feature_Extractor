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
import time

warnings.simplefilter(action='ignore', category=FutureWarning)


# read config file
config_object = ConfigParser()
config_object.read("config.ini")


def progress_bar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar,
    taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)

    # Progress Bar Printing Function
    def print_progress_bar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Initial Call
    print_progress_bar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        print_progress_bar(i + 1)
    # Print New Line on Complete
    print()


def compare_n_compute(df, candidate_pairs, args, logger):
    """
    This function initiates the comparing algorithm and calculates the features. it returns the extracted features
    dataframe.
    """
    dfs_list = []

    for dict_item in progress_bar(args.compare_by, prefix='Progress:', suffix='Complete', length=50):
        for col, algo_list in dict_item.items():
            logger.info(f'Computing for column {col}')

            for al in algo_list:
                compare = rl.Compare()
                compare.string(col, col, al, label=f'{col}_{al}')
                dfs_list.append(compare.compute(candidate_pairs, df))

            time.sleep(0.1)

    logger.info('Finished computing')
    return pd.concat(dfs_list, axis=1)


def index_df(df, args, logger):
    """
    This function gets the candidate pairs to be compared according to the input argument index_by
    """

    indexer = rl.Index()

    for dict_item in args.index_by:
        for method, col_list in dict_item.items():
            if method in ['Block', 'SortedNeighbourhood']:
                for col in col_list:
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

    # lower case the columns we'll compare on
    for dict_item in args.compare_by:
        for col, algo_list in dict_item.items():
            prcs_df[col] = clean(prcs_df[col])

    # remove duplicates if requested
    if args.remove_duplicates:

        # sort by sort_cols
        asc = [False for _ in args.sort_cols]
        prcs_df.sort_values(args.sort_cols, ascending=asc, inplace=True)

        # remove duplicates by drop_cols
        prcs_df.drop_duplicates(subset=args.drop_cols, inplace=True)

    logger.info('Preprocessed the raw dataset')
    return prcs_df


def val_data(raw_df, args, logger):
    """
    This function validates that the columns names in the input data appears in the raw dataset and that the input
    algorithms are supported.
    """

    # validate sort_cols, drop_cols
    for input_col in set(args.sort_cols + args.drop_cols):
        if input_col not in raw_df.columns:
            logger.debug('The column(s) used as input in the parameter \'sort_cols\' or \'drop_cols\' doesn\'t '
                         'match the dataset')
            exit()

    # validate index_by
    for item_dict in args.index_by:
        for method, col_list in item_dict.items():
            if method in ['Block', 'SortedNeighbourhood']:
                for col in col_list:
                    if col not in raw_df.columns:
                        logger.debug('The column(s) used as input in the parameter \'index_by\' doesn\'t '
                                     'match the dataset')
                        exit()

    # validate compare_by
    for item_dict in args.compare_by:
        for col, algo_list in item_dict.items():
            if col not in raw_df.columns:
                logger.debug('The column(s) used as input in the parameter \'compare_by\' doesn\'t match the dataset')
                exit()

            for algo in algo_list:
                if algo not in config_object['const']['supported_algorithms'].split():
                    logger.debug('An algorithm used as input isn\'t supported')
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
            filepath = folder_path + '/' + args.city.lower().replace('_', ' ') + '_' + supplier + '.csv'
            if os.path.exists(filepath):
                t_df = pd.read_csv(filepath, encoding='UTF-8')
                t_df['data_source'] = supplier
                df_list.append(t_df)
    else:
        logger.debug(f'The folder {folder_path} doesn\'t exists, terminating program')
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
                           type=lambda arg: {pair.split(':')[0]: pair.split(':')[1].split(',') for pair in arg.split()},
                           nargs='+', help=config_object["params"]["index_by_help"])

    my_parser.add_argument('--compare_by', '-cb',
                           default=config_object['params']['compare_by'],
                           type=lambda arg: {pair.split(':')[0]: pair.split(':')[1].split(',') for pair in arg.split()},
                           nargs='+', help=config_object["params"]["compare_by_help"])

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
    file_handler = logging.FileHandler('Feature_Extractor_log.txt')
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
    args.city = '_'.join(args.city)
    if isinstance(args.index_by, dict):
        args.index_by = [args.index_by]
    logger.info(f'args={args}')

    # get the data
    raw_df = get_data(args, logger)
    save_path = config_object['const']['data_path_prefix'] + args.city + '/raw_df.csv'
    raw_df.to_csv(save_path)

    # validate the data match input parameters
    val_data(raw_df, args, logger)

    # preprocess the data
    df = preprocess_raw_df(raw_df, args, logger)
    save_path = config_object['const']['data_path_prefix'] + args.city + '/df.csv'
    df.to_csv(save_path)
    # get candidate pairs
    candidate_pairs = index_df(df, args, logger)

    # initialize the compare object and compute the features
    print('Computing..')
    features = compare_n_compute(df, candidate_pairs, args, logger)

    # save the df
    save_path = config_object['const']['data_path_prefix'] + args.city + '/features.csv'
    features.to_csv(save_path)
    logger.info('FINISHED RUNNING')
    print(f'Done, the results can be found at {save_path}')


if __name__ == '__main__':
    main()
