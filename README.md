# distributedSGD
This is a Spark implementation (Python API) for Matrix Factorization using 
Distributed Stochastic Gradient Descent based on : "Large-Scale Matrix 
Factorization with Distributed Stochastic Gradient Descent", Gemulla et al, 
KDD 2011.

## Dependencies
Python 2.7, Numpy, Spark 1.3.0

## Instructions
Using submit script (Ensure your PATH points to the spark bin directory)

spark-submit dsgd\_mf.py `<num_factors>` `<num_workers>` `<num_iterations>`
`<beta_value>` `<lambda_value>`
`<inputV_filepath>` `<outputW_filepath>` `<outputH_filepath>`

## Input
The contents of the input file should be in the following triples format:

`<user_1>`,`<movie_1>`,`<rating_11>`

...

`<user_i>`,`<movie_j>`,`<rating_ij>`

...

`<user_M>`,`<movie_N>`,`<rating_ij>`

Where, user i is an integer ID for the ith user, movie j is an integer ID for 
the jth movie, and rating ij is the rating given by user i to movie j.

For example, 

1,3114,4

1,608,4

1,1246,4

2,1357,5

2,3068,4

2,1537,4

...

6040,562,5

6040,1096,4

6040,1097,4

## Output
Factor matrices are written to the specified output files in non-sparse, csv
format. An example 4 X 5 matrix:

3.0,6,9,12,15

5,10,15.2,20,25

7,14,21.5,28,35

11,22.2,33,44.8,5

## Running on an AWS cluster
* Launch an AWS EMR cluster with Spark preinstalled. Copy the input data file
to an S3 bucket and make it public.
* Log into the master node and run the ../spark/sbin/start-master script. Note
down the master IP and port from the generated log file.
* Log into each of the slave machines in the cluster and run the following script:
../spark/bin/spark-class org.apache.spark.deploy.worker.Worker 
spark://`<Ip address of master>:<port of master>`. This will set up the cluster
and ensure slaves can connect to the master.
* In the master, run the job :
../spark/bin/spark-submit --master `spark://<Ip address of master>:<port of master>` 
dsgd\_mf.py `<num_factors>` `<num_workers>` `<num_iterations>`
`<beta_value>` `<lambda_value>`
`<inputV_filepath>` `<outputW_filepath>` `<outputH_filepath>`
* In the above,
`<inputV_filepath = s3n://<bucket-name>/../input_file.csv`
