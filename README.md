# Exact Resampling-based MapReduce Particle Filtering

This is an improved MapReduce implementation of the particle filter using the systematic resampling,
also known as minimum variance resampling. The proposed resampling strategy is with the parallel complexity
of O( (log(n))^2) instead of the original O( (logn)^3), where n is the particles size.

### Guide to run the code
These instructions will allow you to run the particle filter Hadoop and Spark source code.

> #### Generate Data
> The initial input on both Hadoop and Spark (for large input sizes) is provided via a .txt file.
> To generate an input file use the C++ code under the ```data_gen/``` directory. Compile and execute to produce the input file : 
> ```ubuntu
> g++ data.cpp
> ./a.out
> ```

> #### Hadoop
> The assumption is that Hadoop is already installed in [pseudo-distributed mode](http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html). <br/><br/>
> Initial steps
>> 1. Create a directory with name ```input0/```
>> 2. Copy under the ```input0/``` the file generated from the [previous step](https://github.com/particlefilter/mrpf#generate-data).
>> 3. Create the ```myjar.jar``` file. We create the ```.jar``` file using the [Eclipse IDE](https://eclipse.org/downloads/).

> Final Steps
>> ```hadoop
>> hadoop namenode -format
>> start-all.sh
>> hadoop fs -mkdir -p /user/username
>> hadoop fs -put input0
>> hadoop jar myjar.jar PF_Hadoop input0 output 1024
>> or
>> nohup hadoop jar myjar.jar PF_Hadoop input0 output 1024 &

> #### Spark
> The assumption is that Apache Spark is installed in [standalone mode](http://spark.apache.org/docs/latest/spark-standalone.html#spark-standalone-mode) or [cluster mode](http://spark.apache.org/docs/latest/cluster-overview.html) with the Hadoop YARN  cluster manager
>  1. Setup the [sbt tool](http://spark.apache.org/docs/latest/quick-start.html#self-contained-applications)
>  2. Copy the .txt file to HDFS : ```hadoop fs -put in2_10.txt```
>  3. Compile the code :  ```sbt package```
>  4. Run the code : ```sbt run``` or ```spark-submit –master="yarn"  –num-executors=8 –executor-cores=10 PF_Spark.scala```. Use ```nohup``` similarly with Hadoop to
> run the command into the background.
>  Depending the installation it is required to configure the Spark properties via the [Spark configuration](http://spark.apache.org/docs/latest/configuration.html#spark-configuration).
### Contents

- src/ : Contains 
  - src/GPF_Hadoop.java : Hadoop version of the generic particle filter.
  - src/GPF_Spark.scala : Spark version of the generic particle filter.
  - data_gen/ contain the source file
    - data.cpp : Generate an input file	

### Authors

- Lykourgos Kekempanos
- Thiyagalingam Jeyarajan
- Simon Maskell
  
### License

Copyright(c)2016 particlefilter, University of Liverpool

This project is licensed under the MIT License - see [LICENSE](https://github.com/particlefilter/mrpf/blob/master/LICENSE) for details

### Acknowledgments

- STFC Daresbury and Hartree Center
- UK EPSRC Doctoral Training Award
