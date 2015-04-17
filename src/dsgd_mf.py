#!/usr/bin/env python
__author__ = 'abhishek'


import argparse
import os
from pyspark import SparkContext, SparkConf
import numpy as np


def parseLine(inputString):
    lines = inputString.split(',')
    return (int(lines[0]), int(lines[1]), float(lines[2]))

def sgd(V, W, H, workerId):
    Wmapped = {int(key):value for (key, value) in [(elem[0], elem[1]) for elem in W]}
    Hmapped = {int(key):value for (key, value) in [(elem[0], elem[1]) for elem in H]}
    nprime = 0
    n = global_updates_b.value
    for rating in V:
        vij = rating[2]
        Wi = Wmapped[rating[0]]
        Hj = Hmapped[rating[1]]
        whProd = np.dot(Wi, Hj)
        factor = (whProd - vij)*2
        Wloss = factor*Hj + (2*LAMBDA/Ni[rating[0]]) * Wi
        Hloss = factor*Wi + (2*LAMBDA/Nj[rating[1]]) * Hj
        epsilon = pow((n + nprime + tau.value), -beta.value)
        Wprime = Wi - epsilon*Wloss
        Hprime = Hj - epsilon*Hloss
        Wmapped[rating[0]] = Wprime
        Hmapped[rating[1]] = Hprime
        nprime += 1
    # update epsilon globally
    global_updates_a.add(nprime)
    return Wmapped, Hmapped

def myMapFunc(iterator):
    result = []
    for elem in iterator:
        workerId = elem[0]
        V = sorted(list([item for item in elem[1][0] if (item[1]-1)%NUM_WORKERS == (workerId + stratum)%NUM_WORKERS]))  # Extract and filter V
        W = sorted(list(elem[1][1]))                                                                         # Extract W
        H = sorted(list(elem[1][2]))                                                                         # Extract H
        Wnew, Hnew = sgd(V, W, H, workerId)
        yield (Wnew, Hnew)


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
    parser.add_argument('input_filepath', help='file containing training data - ratings data', type=str)
    parser.add_argument('outputW_filepath', help='output file for parameter W', type=str)
    parser.add_argument('outputH_filepath', help='output file for parameter H transpose', type=str)
    args = parser.parse_args()

    FACTORS = args.num_factors
    NUM_WORKERS = args.num_workers
    ITERATIONS = args.num_iterations
    BETA = args.beta_value
    LAMBDA = args.lambda_value
    INPUTFILE = args.input_filepath
    WFILE = args.outputW_filepath
    HFILE = args.outputH_filepath
    TAU = 100

    # Read the ratings matrix
    sc = SparkContext("local", "dsgd")
    vTuples = sc.textFile(INPUTFILE).map(lambda line: parseLine(line))

    NUM_COLS = vTuples.map(lambda x : x[1]).max()
    NUM_ROWS = vTuples.map(lambda x : x[0]).max()

    # Create and initialize W and H matrices
    Wcurr = np.random.ranf(size=(NUM_ROWS, FACTORS))                    # Num of Users X Factors = W
    Hcurr = np.random.ranf(size=(NUM_COLS, FACTORS))                    # Num of Users X Factors = H'

    NUM_STRATA = NUM_WORKERS                                             # Number of strata

    # Build keys for W
    rowIndices = range(1, NUM_ROWS+1)
    colIndices = range(1, NUM_COLS+1)

    # Compute and broadcast Ni and Nj
    Ni = vTuples.keyBy(lambda x : x[0]).map(lambda x : (x[0], 1)).reduceByKey(lambda a, b : a + b).collectAsMap()
    Nj = vTuples.keyBy(lambda x : x[1]).map(lambda x : (x[0], 1)).reduceByKey(lambda a, b : a + b).collectAsMap()

    # Broadcast TAU and BETA, we didn't know why we do this, perhaps it works by just being in scope
    tau = sc.broadcast(TAU)
    beta = sc.broadcast(BETA)
    #f = open('output', 'w')
    Wpaired = zip(rowIndices, Wcurr)                 # list
    Hpaired = zip(colIndices, Hcurr)

    accumulated = 0
    global_updates_a = sc.accumulator(0)

    for epoch in range(1, ITERATIONS+1):
        global_updates_b = sc.broadcast(accumulated)
        for stratum in range(NUM_STRATA):
            # Build keyed version of W
            W_keyed = sc.parallelize(Wpaired).keyBy(lambda x : (x[0]-1) % NUM_WORKERS).partitionBy(NUM_WORKERS)     # perhaps optimize here more
            # Build keyed version of H
            H_keyed = sc.parallelize(Hpaired).keyBy(lambda x : (NUM_WORKERS+((x[0]-1)%NUM_WORKERS)-stratum) % NUM_WORKERS).partitionBy(NUM_WORKERS)
            # Build keyed version of V (only based on row, columns will be filtered out in the map function)
            V_keyed = vTuples.keyBy(lambda x : (x[0]-1) % NUM_WORKERS).partitionBy(NUM_WORKERS)                  # perhaps optimize here more
            # Group W, H and V
            result = list(V_keyed.groupWith(W_keyed, H_keyed).mapPartitions(myMapFunc).collect())
            # list of NUM_WORKERS lists, each containing of 1 tuple of 2 elements -> (index, row/column as list of values)
            # accumulate all the results into Wpaired and Hpaired
            Wmapped = {}
            Hmapped = {}
            for res in result:
                Wmapped.update(res[0])
                Hmapped.update(res[1])
            Wpaired = Wmapped.items()      #  list of tuples
            Hpaired = Hmapped.items()      #  list of tuples
        accumulated = global_updates_a.value
        global_updates_b.unpersist()

    # Write out W
    W = np.matrix([row[1] for row in sorted(Wpaired, key = lambda tup : tup[0])])
    f = open(WFILE, 'w')
    for row in W:
        np.savetxt(f, row, fmt="%.1f", delimiter=',')
    f.close()
    with open(WFILE, 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()

    # Write out H after transpose
    H_transpose = np.matrix([col[1] for col in sorted(Hpaired, key = lambda tup : tup[0])]).transpose()
    f = open(HFILE, 'w')
    for col in H_transpose:
        np.savetxt(f, col, fmt="%.1f", delimiter=',')
    f.close()
    with open(HFILE, 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()









