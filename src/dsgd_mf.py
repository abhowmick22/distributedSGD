#!/usr/bin/env python
__author__ = 'abhishek'

import argparse
import os
from pyspark import SparkContext, SparkConf
import numpy as np
from scipy import sparse
import csv

# load a factor matrix from non-sparse, csv file
def LoadMatrix(csvfile):
	data = np.genfromtxt(csvfile, delimiter=',')
	return np.matrix(data)

# load training matrix from sparse, csv file
def LoadSparseMatrix(csvfile):
        val = []
        row = []
        col = []
        select = []
        f = open(csvfile)
        reader = csv.reader(f)
        for line in reader:
                row.append( int(line[0])-1 )
                col.append( int(line[1])-1 )
                val.append( int(line[2]) )
                select.append( (int(line[0])-1, int(line[1])-1) )
        return sparse.csr_matrix( (val, (row, col)) ), select

# compute reconstruction error
def CalculateError(V, W, H, select):
        diff = V-W*H
        error = 0
        for row, col in select:
                error += diff[row, col]*diff[row, col]
        return error/len(select)

def writeResults(WFILE, HFILE, Wpaired, Hpaired):
    W = np.matrix([row[1] for row in sorted(Wpaired, key = lambda tup : tup[0])])
    f = open(WFILE, 'w')
    for row in W:
        np.savetxt(f, row, fmt="%.1f", delimiter=',')
    f.close()
    # remove last newline character in file
    with open(WFILE, 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()

    # Write out H' after transposing
    H_transpose = np.matrix([col[1] for col in sorted(Hpaired, key = lambda tup : tup[0])]).transpose()
    f = open(HFILE, 'w')
    for col in H_transpose:
        np.savetxt(f, col, fmt="%.1f", delimiter=',')
    f.close()
    # remove last newline character in file
    with open(HFILE, 'rb+') as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()

# function to extract rating tuples from the training data file
def parseLine(inputString):
    lines = inputString.split(',')
    return (int(lines[0]), int(lines[1]), float(lines[2]))

# function to perform SGD updates
def sgd(V, W, H):
    Wmapped = {int(key):value for (key, value) in [(elem[0], elem[1]) for elem in W]}
    Hmapped = {int(key):value for (key, value) in [(elem[0], elem[1]) for elem in H]}
    nprime = 0                                                                           # Num SGD updates
    n = global_updates_b.value                                                           # updates performed before this iteration
    for rating in V:
        vij = rating[2]
        w_i = Wmapped[rating[0]]
        h_j = Hmapped[rating[1]]
        whProd = np.dot(w_i, h_j)
        factor = (whProd - vij)*2
        w_loss = factor*h_j + (2*LAMBDA/Ni[rating[0]]) * w_i
        h_loss = factor*w_i + (2*LAMBDA/Nj[rating[1]]) * h_j
        epsilon = pow((n + nprime + tau.value), -beta.value)
        w_prime = w_i - epsilon*w_loss
        h_prime = h_j - epsilon*h_loss
        Wmapped[rating[0]] = w_prime
        Hmapped[rating[1]] = h_prime
        nprime += 1
    # add num of SGD updates to global counter
    global_updates_a.add(nprime)
    return Wmapped, Hmapped

# map function to be applied to each partition
def myMapFunc(iterator):
    result = []
    for elem in iterator:
        workerId = elem[0]
        V = list([item for item in elem[1][0] if
                         (item[1]-1)%NUM_WORKERS == (workerId + stratum)%NUM_WORKERS])               # Extract and filter V
        W = list(elem[1][1])                                                                         # Extract W
        H = list(elem[1][2])                                                                         # Extract H
        Wnew, Hnew = sgd(V, W, H)
        yield (Wnew, Hnew)


if __name__ == "__main__":
    '''
    dsgd_mf.py <num_factors> <num_workers> <num_iterations> \
    <beta_value> <lambda_value> \
    <inputV_filepath> <outputW_filepath> <outputH_filepath>
    '''
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

    # Read the ratings matrix into an RDD
    sc = SparkContext("local", "dsgd")
    vTuples = sc.textFile(INPUTFILE).map(lambda line: parseLine(line))

    # Determine dimensions of ratings matrix
    NUM_COLS = vTuples.map(lambda x : x[1]).max()
    NUM_ROWS = vTuples.map(lambda x : x[0]).max()
    NUM_STRATA = NUM_WORKERS                                            # Number of strata

    # Create and initialize factor matrices
    Wcurr = np.random.ranf(size=(NUM_ROWS, FACTORS))                    # Num of Users X Factors = W
    Hcurr = np.random.ranf(size=(NUM_COLS, FACTORS))                    # Num of Users X Factors = H'

    # Build keys for W
    rowIndices = range(1, NUM_ROWS+1)
    colIndices = range(1, NUM_COLS+1)

    # Compute and broadcast Ni*/N*j, ie num of non zero elements per row/column in rating matrix
    Ni = vTuples.keyBy(lambda x : x[0]).map(lambda x : (x[0], 1)).reduceByKey(lambda a, b : a + b).collectAsMap()
    Nj = vTuples.keyBy(lambda x : x[1]).map(lambda x : (x[0], 1)).reduceByKey(lambda a, b : a + b).collectAsMap()

    # Broadcast TAU and BETA, we didn't know why we do this, perhaps it works by just being in scope
    tau = sc.broadcast(TAU)
    beta = sc.broadcast(BETA)

    # Pair factor matrices with their indices, this is useful later for recurring updates
    Wpaired = zip(rowIndices, Wcurr)
    Hpaired = zip(colIndices, Hcurr)

    # accumulator variables to keep track of SGD updates across iterations
    accumulated = 0
    global_updates_a = sc.accumulator(0)

    for epoch in range(1, ITERATIONS+1):
        global_updates_b = sc.broadcast(accumulated)
        for stratum in range(NUM_STRATA):
            # Build keyed version of W, perhaps we can optimize by persisting these partitioned RDDs on local workers
            W_keyed = sc.parallelize(Wpaired).keyBy(lambda x : (x[0]-1) % NUM_WORKERS).partitionBy(NUM_WORKERS)
            # Build keyed version of H
            H_keyed = sc.parallelize(Hpaired).keyBy(lambda x : (NUM_WORKERS+((x[0]-1)%NUM_WORKERS)-stratum) % NUM_WORKERS).partitionBy(NUM_WORKERS)
            # Build keyed version of V, perhaps we can optimize by persisting these partitioned RDDs on local workers
            V_keyed = vTuples.keyBy(lambda x : (x[0]-1) % NUM_WORKERS).partitionBy(NUM_WORKERS)
            # Group W, H and V
            result = list(V_keyed.groupWith(W_keyed, H_keyed).mapPartitions(myMapFunc).collect())
            # accumulate all the results from worker nodes
            Wmapped = {}
            Hmapped = {}
            for res in result:
                Wmapped.update(res[0])
                Hmapped.update(res[1])
            Wpaired = Wmapped.items()
            Hpaired = Hmapped.items()
        accumulated = global_updates_a.value
        global_updates_b.unpersist()

        # write results and compute reconstruction error after every iteration, this should be disabled for actual run
        '''
        writeResults(WFILE, HFILE, Wpaired, Hpaired)
        W = LoadMatrix(WFILE)
        H = LoadMatrix(HFILE)
        V, select = LoadSparseMatrix(INPUTFILE)
        error = CalculateError(V,W,H,select)
        print error
        '''

    # Write out final results
    writeResults(WFILE, HFILE, Wpaired, Hpaired)
    W = LoadMatrix(WFILE)
    H = LoadMatrix(HFILE)
    V, select = LoadSparseMatrix(INPUTFILE)
    error = CalculateError(V,W,H,select)
    print 'Reconstruction error is ', error