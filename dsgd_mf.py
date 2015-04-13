__author__ = 'abhishek'

import argparse


if __name__ == "__main__":
    '''
    dsgd_mf.py <num_factors> <num_workers> <num_iterations> \
    <beta_value> <lambda_value> \
    <inputV_filepath> <outputW_filepath> <outputH_filepath>
    '''

    # BASE for data -- disable when submitting to Autolab
    BASE = 'data/'

    # read in params
    parser = argparse.ArgumentParser()
    parser.add_argument('num_factors', help='the rank for decomposition')
    parser.add_argument('num_workers', help='number of workers to spawn')
    parser.add_argument('num_iterations', help='number of iterations for experimental evaluation', type=int)
    parser.add_argument('beta_value', help='beta parameter', type=float)
    parser.add_argument('lambda_value', help='regularization parameter', type=float)
    parser.add_argument('inputV_filepath', help='file containing training matrix - ratings data', type=str)
    parser.add_argument('outputW_filepath', help='output file for parameter W', type=str)
    parser.add_argument('outputH_filepath', help='output file for parameter H transpose', type=str)
    args = parser.parse_args()

    factors = args.num_factors
    workers = args.num_workers
    iterations = args.num_iterations
    beta_param = args.beta_value
    lambda_param = args.lambda_value
    inputV = args.inputV_filepath
    outputW = args.outputW_filepath
    outputH = args.outputH_filepath

    

