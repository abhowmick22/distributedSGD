__author__ = 'abhishek'

import argparse
from pyspark import SparkContext, SparkConf
import numpy as np


def parseLine(inputString):
    lines = inputString.split(',')
    return (int(lines[0]), int(lines[1]), float(lines[2]))

def getNumTuples(input):
    return len(input)

def sgd(V, W, H):
    f = open('output', 'w')
    f.write("came herexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    epsilon = pow((epoch + tau.value), -beta.value)
    Wmapped = {int(key):value for (key, value) in [(elem[0], elem[1]) for elem in W]}
    f.write('Wmap')
    #f.write(str(Wmapped))
    f.write("\n")
    Hmapped = {int(key):value for (key, value) in [(elem[0], elem[1]) for elem in H]}
    #f.write('Hmap')
    f.write(str(Hmapped))
    f.write("\n")
    Wnew = []
    Hnew = []
    for rating in V:
        vij = rating[2]
        Wi = Wmapped[rating[0]]
        Hj = Hmapped[rating[1]]
        whProd = np.dot(Wi, Hj)
        factor = (whProd - vij)*2
        Wloss = factor*Hj
        Hloss = factor*Wi
        Wprime = Wi - epsilon*Wloss + (2*LAMBDA/Ni[rating[0]] * Wi)
        Hprime = Hj - epsilon*Hloss + (2*LAMBDA/Nj[rating[1]] * Hj)
        Wnew.append((rating[0], Wprime))
        Hnew.append((rating[1], Hprime))
    f.flush()
    f.close()
    return Wnew, Hnew

def myMapFunc(iterator):
    result = []
    for elem in iterator:
        workerId = elem[0]
        V = [item for item in elem[1][0] if (item[1] + stratum)%NUM_WORKERS == workerId]        # Extract and filter V
        W = elem[1][1]                                                                          # Extract W
        H = elem[1][2]                                                                          # Extract H
        Wnew, Hnew = sgd(V, W, H)
        result.append((Wnew, Hnew))
    yield iter(result)



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
    parser.add_argument('num_factors', help='the rank for decomposition', type=int)
    parser.add_argument('num_workers', help='number of workers to spawn', type=int)
    parser.add_argument('num_iterations', help='number of iterations for experimental evaluation', type=int)
    parser.add_argument('beta_value', help='beta parameter', type=float)
    parser.add_argument('lambda_value', help='regularization parameter', type=float)
    parser.add_argument('test_filepath', help='file containing test data - ratings data', type=str)
    parser.add_argument('outputW_filepath', help='output file for parameter W', type=str)
    parser.add_argument('outputH_filepath', help='output file for parameter H transpose', type=str)
    args = parser.parse_args()

    FACTORS = args.num_factors
    NUM_WORKERS = args.num_workers
    ITERATIONS = args.num_iterations
    BETA = args.beta_value
    LAMBDA = args.lambda_value
    TESTFILE = args.test_filepath
    WFILE = args.outputW_filepath
    HFILE = args.outputH_filepath
    TAU = 100

    # Read the ratings matrix
    inputFile = DATA_BASE + 'small_input/nf_subsample.csv'
    sc = SparkContext("local", "dsgd")
    vTuples = sc.textFile(inputFile).map(lambda line: parseLine(line))

    NUM_COLS = vTuples.map(lambda x : x[1]).max()
    NUM_ROWS = vTuples.map(lambda x : x[0]).max()

    # Create and initialize W and H matrices
    Wcurr = np.random.ranf(size=(NUM_ROWS, FACTORS))*10.0                # Num of Users X Factors = W
    Hcurr = np.random.ranf(size=(NUM_COLS, FACTORS))*10.0                # Num of Users X Factors = H'

    # TODO : Update NUM_STRATA
    NUM_STRATA = NUM_WORKERS                                             # Number of strata

    # Build keys for W
    rowIndices = sc.parallelize(range(1, NUM_ROWS+1))
    colIndices = sc.parallelize(range(1, NUM_COLS+1))

    # Compute and broadcast Ni and Nj
    Ni = vTuples.keyBy(lambda x : x[0]).map(lambda x : (x[0], 1)).reduceByKey(lambda a, b : a + b).collectAsMap()
    Nj = vTuples.keyBy(lambda x : x[1]).map(lambda x : (x[0], 1)).reduceByKey(lambda a, b : a + b).collectAsMap()

    # Broadcast TAU and BETA, we didn't know why we do this, perhaps it works by just being in scope
    tau = sc.broadcast(TAU)
    beta = sc.broadcast(BETA)

    for epoch in range(1, ITERATIONS+1):
        for stratum in range(NUM_STRATA):
            # Build keyed version of W
            W_keyed = rowIndices.zip(sc.parallelize(Wcurr)).keyBy(lambda x : x[0] % NUM_WORKERS).partitionBy(NUM_WORKERS)
            # Build keyed version of H
            H_keyed = colIndices.zip(sc.parallelize(Hcurr)).keyBy(lambda x : (x[0]+stratum) % NUM_WORKERS).partitionBy(NUM_WORKERS)
            # Build keyed version of V (only based on row, columns will be filtered out in the map function)
            V_keyed = vTuples.keyBy(lambda x : x[0] % NUM_WORKERS).partitionBy(NUM_WORKERS)
            # Group W, H and V
            result = V_keyed.groupWith(W_keyed, H_keyed).mapPartitions(myMapFunc).collectAsMap()











