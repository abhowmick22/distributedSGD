__author__ = 'abhishek'

import argparse
from pyspark import SparkContext, SparkConf


def parseFile(inputString):
    lines = inputString.split()
    movieId = lines[0]
    reviews = lines[1:]
    tuples = []
    for rev in reviews :
        fields = rev.split(',')
        tuples.append((fields[0], movieId, fields[1]))    # (movieId, userId, rating) triple
    return tuples


if __name__ == "__main__":
    '''
    dsgd_mf.py <num_factors> <num_workers> <num_iterations> \
    <beta_value> <lambda_value> \
    <inputV_filepath> <outputW_filepath> <outputH_filepath>
    '''

    # BASE for data -- disable when submitting to Autolab
    DATA_BASE = 'data/'

    # read in params
    parser = argparse.ArgumentParser()
    parser.add_argument('num_factors', help='the rank for decomposition')
    parser.add_argument('num_workers', help='number of workers to spawn')
    parser.add_argument('num_iterations', help='number of iterations for experimental evaluation', type=int)
    parser.add_argument('beta_value', help='beta parameter', type=float)
    parser.add_argument('lambda_value', help='regularization parameter', type=float)
    parser.add_argument('test_filepath', help='file containing test data - ratings data', type=str)
    parser.add_argument('outputW_filepath', help='output file for parameter W', type=str)
    parser.add_argument('outputH_filepath', help='output file for parameter H transpose', type=str)
    args = parser.parse_args()

    factors = args.num_factors
    workers = args.num_workers
    iterations = args.num_iterations
    beta_param = args.beta_value
    lambda_param = args.lambda_value
    testFile = args.test_filepath
    outputW = args.outputW_filepath
    outputH = args.outputH_filepath

    # Read the ratings matrix
    inputFile = DATA_BASE + 'training_set'
    sc = SparkContext("local", "dsgd")
    # get a list of tuples (userId, movieId, rating)
    vTuples = sc.wholeTextFiles(inputFile).flatMap(lambda entry: parseFile(entry[1])).collect()

    print(vTuples)


