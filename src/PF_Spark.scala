/***********************************************************
*                Particle filter                           *
* Details : https://github.com/particlefilter/mrpf/        *
* Copyright(c)2016 University of Liverpool                 *
************************************************************/

import org.apache.spark.{SparkContext,SparkConf}
import math.{ceil,floor,log,pow,sqrt,cos,Pi,exp,abs,log1p,max}
import org.apache.spark.rdd.RDD
import scala.util.Random
import java.util.concurrent.ThreadLocalRandom
import scala.util.control.Breaks._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.storage.StorageLevel._

object PF_Spark {
  def main(args: Array[String]) {
    //start time, can be ignored
    val t0 = System.nanoTime()
    
    //basic configuration local mode
    val conf = new SparkConf()
          .setMaster("local[2]")
          .setAppName("Particle Filter")
    
    val sc = new SparkContext(conf)
    
    //number of particles. must be power of 2
    val Nlength:Int = pow(2,10).toInt
    
    //exponent of particles input size
    val NumIter:Int = (log(Nlength)/log(2)).toInt;
    
    //Initialisation
    val Qk_1 = sc.broadcast(10)        //variance of the system noise
    val Rk   = sc.broadcast(1)         //variance of the measurement noise
    val time:Int    = 5                //number of iterations
    val Nt:Float   =  0.50f*Nlength    //effective sample size threshold
    var x:Float       = 1.0f           //initial actual value
    var z:Float       = 1.0f           //measurement value
    val initVar:Float = 10.0f          //variance of the initial estimate
    
    //STEP 1: Create Particles
    var vec_Pardd = sc.textFile("in2_10")
                      .map(x => {val t= x.split(","); (t(0).toInt -> t(1).toFloat) })
    
    //random values
    val rand = ThreadLocalRandom.current()
    
    //particles initialisation
    vec_Pardd = vec_Pardd.mapValues( x => x.toFloat + (sqrt(initVar).toFloat)*rand.nextGaussian.toFloat)
                         .persist(MEMORY_AND_DISK)
    
    for (t <- 1 to time){
      //update the actual system and the measurement equation
      x = (0.5f*x + 25.0f*x/(1+x*x) + 8.0f*cos(1.2f*(t-1)) + sqrt(Qk_1.value)*rand.nextGaussian).toFloat
      
      //exponent of particles input size
      val NumIter:Int = (log(Nlength)/log(2)).toInt;
      
      //update the actual measurement equation
      val z = sc.broadcast(((x*x)/20.0 + sqrt(Rk.value)*rand.nextGaussian).toFloat)
      
      //STEP 2: Importance Weights
      
      //state update
      val new_staterdd = vec_Pardd.mapValues( x => (0.5*x.toFloat + 25.0*x/(1.0+x*x) + 8.0*cos(1.2*(t-1)) + (sqrt(Qk_1.value))*rand.nextGaussian).toFloat)
      
      //measurement update
      val new_measurdd = new_staterdd.mapValues( x =>  (x*x/20.0).toFloat)
      
      //assign weight to each particle
      var vecl_weightsrdd = new_measurdd.mapValues( x =>  (-log(sqrt(2*Pi*Rk.value)) - ((z.value - x)*(z.value - x))/(2*Rk.value)).toFloat)
      
      //weights normalisation
      vecl_weightsrdd = WeightsNormalization(vecl_weightsrdd, NumIter, Nlength)
      
      //STEP 3: Resampling (if needed)
	    
      //effective sample size
      val Neff = EffectiveSampleSize(vecl_weightsrdd, NumIter)
      
      if (Neff <= Nt){
        println("Run Resampling")
        
        //convert weights to non-logarithmic form
        vecl_weightsrdd = vecl_weightsrdd.mapValues(x =>  exp(x).toFloat)
        
        //Step A : Minimum variance resampling - produce number of copies for each particle 
        val Ncopies:RDD[(Int,Int)] = minvarresample(vecl_weightsrdd, NumIter, Nlength, sc)
        
        //Step B : Sort the number of copies in descending order
        val bsout: RDD[(Int, (Int, Float))] = bitonicSort(Ncopies, new_staterdd, NumIter, Nlength)
        
        //Step C : Redistribution - produce the new population of particles
        //algorithm steps
        val maxIter:Int = NumIter - 1
        val leafs:Int = maxIter
        
        val redout = redistribution(bsout, Nlength, leafs, sc)
        
        //Step D : Mean value estimation
        val xest:RDD[Float] = xestimate(redout, Nlength, NumIter)
        
        //print real/estimated value
        println("Iteration        = " + t)
        println("real value       = " + x)
        println("estimated value  = " + xest.first())
        println("------------------------------------")
        
        //particles update
        vec_Pardd = redout
        
      }else{
        //Mean value estimation
        val xest:RDD[Float] = xestimate(new_staterdd, Nlength, NumIter)
        
        //print real/estimated value
        println("Iteration        = " + t)
        println("real value       = " + x)
        println("estimated value  = " + xest.first())
        println("------------------------------------")
        
        //particles update
        vec_Pardd = new_staterdd
      }
    }
    
    //end time
    val t1 = System.nanoTime()
    println("==============================")
    println("Total Time                    : " + (t1-t0)/1000000000.0 +  " sec")
    println("PROGRAM_END")
    
    sc.stop
  } //end_main
  
  
  /***********************************
   *  Estimate Mean Value of Posterior
   ***********************************/
  
  def xestimate(vec_Pardd: RDD[(Int,Float)], Nlength: Int, NumIter: Int) : RDD[Float] = {
    var vec_ParddSum:RDD[(Int, Float)] = vec_Pardd
    
    //compute sum(vec_Particles)
    for(i <- 1 to NumIter){
        vec_ParddSum =  vec_ParddSum.map(x => (ceil(x._1/2.0).toInt -> x._2))
                                    .reduceByKey(_+_)
    }
    
    //particles mean estimation
    val xest:RDD[Float] = vec_ParddSum.map(x => x._2/Nlength)
    
    (xest)
  }
  
  /***************************
   *  Redistribute Algorithm
   ***************************/
  
  def redistribution(in:RDD[(Int, (Int, Float))], Nlength:Int, maxIter:Int, sc:SparkContext):RDD[(Int, Float)] = {
    //perform the computations on the old population of particles and the number of copies
    //using a separate vector
    var nco = in.map({case (x,y) => x -> y._1}) //number of copies
    var osa = in.map({case (x,y) => x -> y._2}) //old population of particles
    
    //the method begins from the root of the balanced binary tree (i.e. level 0)
    //and re-executes the method for log(N) steps, where N is the particles size
    val level:Int = 0
    
    val oldsamples = callred(nco, osa, Nlength, maxIter, sc:SparkContext)
    
    oldsamples
  }
  
  def callred(nc: RDD[(Int, Int)], os: RDD[(Int, Float)], Nlength:Int, maxIter:Int, sc:SparkContext):RDD[(Int, Float)] = {
    
    var nco = nc //number of copies
    var osa = os //particles population
    
    //Build the binary tree within the two vectors, 'nco' and 'osa';
    //The root node, level 0, handles all the keys from [1,...,N]
    //In level 1, the left and right nodes handle the keys [1,...,N/2] and N/2+1 to N, respectively.
    //In the leaf nodes, each leaf node handles a single key.
    for(level <- 0 until (maxIter+1)){
        
        val adj = (Nlength/pow(2,level)).toInt
        val NumIter = (log(adj)/log(2.0)).toInt
        
        //compute max nodeID
        val maxNodeID:Int = (Nlength/((Nlength/pow(2,level)))).toInt
        
        val ztmp = new ListBuffer[(Int,Int)]
        ztmp ++=  ListBuffer((0, 0))
        
        val ztmprdd = sc.parallelize(ztmp)
        
        val  zeros = ztmprdd.map(x => for(j <- maxNodeID to 1 by -1) yield (j*adj-adj+j, 0))
                            .flatMap(x => x)
        
        //cN = [0 cumsum(Ncopies)]
        //the cumulative summation computation is achieved across all nodes
        val cNtemp: RDD[(Int, Int)] = cumulativeSummationred(nco, NumIter)
                                      .map( d => {val nodeID:Int = ceil(d._1/(Nlength/pow(2,level))).toInt; (d._1+nodeID -> d._2)})
        
        //add the pairs with the zero values
        val cN = cNtemp.union(zeros).coalesce(ztmprdd.partitions.size)
        
        val mindif: RDD[(Int, Int)] = cN.filter(d => {val zeroval:Int = d._1 % (adj+1); zeroval <= (adj/2 + 1) && zeroval != 0 })
                                        .map( d => {val nkey = d._1 - (ceil(d._1/(adj+1)).toInt - 1)-1; 
                                                    if(d._2 <= (adj/2)) (nkey -> d._2) else (nkey -> (adj/2))  })
        
	//compute even diff()
        val w1: RDD[(Int, Int)] = mindif.map(x => (x._1 % adj -> (x._1 -> x._2)))
                                        .filter( x => x._1 <= adj/2 && x._1 >= 0 )
                                        .map{ case (x,y) => { (y._1 -> y._2)}}
                                        .map(x => (ceil(x._1/2.0).toInt -> x._2))
                                        .reduceByKey((x,y) =>  math.abs(x-y))
                                        .map(x => if (x._1 == 0) (x._1+1 -> x._2) else (2*x._1-1 -> x._2))
        
        //compute odd diff()
        val w2:RDD[(Int, Int)] = mindif.map(x => (x._1 % adj -> ( x._1 -> x._2)))
                                       .filter( x => x._1 <= (adj/2 + 1) && x._1 > 1 )
                                       .map{ case (x,y) => { (y._1 -> y._2)}}
                                       .map(x => (floor(x._1/2.0).toInt -> x._2))
                                       .reduceByKey((x,y) =>  math.abs(x-y))
                                       .map(x => (2*x._1 -> x._2))
        
        //merge to have the Ncopies
        val Nfirst = w1.union(w2).coalesce(ztmprdd.partitions.size)
        
        val oldsamples_first:RDD[(Int,Float)] = osa.map{ x => {val zeroval = x._1 % adj; 
                                                               if (zeroval <= (adj/2) && zeroval != 0 )  (x._1 -> x._2)
                                                               else (x._1 -> Float.MinValue)}}
                                                     .filter( x => x._2 != Float.MinValue)
        
        //compute Nraw = Ncopies(1:N/2) - Nfirst                           
        val NrawTemp:RDD[(Int,Int)] = nco.map{ x => {val zeroval = x._1 % adj;
                                                     if (zeroval <= (adj/2) && zeroval != 0 )  (x._1 -> x._2)
                                                     else (x._1 -> -1)}}
                                         .filter( x => x._2 != -1)
         
        val Nraw1:RDD[(Int,Int)] = (NrawTemp.union(Nfirst)).coalesce(ztmprdd.partitions.size)
                                            .reduceByKey( (x,y) => abs((x-y)))
        
        //compute nleft = sum(cN(2:(N/2+1))<=N/2)
        //filter to the correct keys
        //nleft1 = cN(2:(N/2+1))
        val nleft1 = cN.map{ x => {val zeroval = x._1 % (adj+1);
                                   if (zeroval <= (adj/2 + 1) && zeroval >= 2 )  (x._1 -> x._2)
                                   else (x._1 -> -1)}}
                       .filter( x => x._2 != -1)
        
        //compute cN(2:(N/2+1))<=N/2
        //nleftemp (initialization) = key | value | nodeID
        var nleftemp = nleft1.map( d => {val nodeID:Int = ceil(d._1/(adj+1.0)).toInt;
                                    if (d._2 <= adj/2)
                                      if (nodeID % 2 == 1)
                                        (d._1-1, 1 -> nodeID)
                                      else
                                        (d._1, 1 -> nodeID)
                                    else
                                      if (nodeID % 2 == 1)
                                        (d._1-1, 0 -> nodeID)
                                      else
                                        (d._1, 0 -> nodeID) })
        
        //compute sum(cN(2:(N/2+1))<=N/2)
        for(i <- 1 to maxIter-level){
          nleftemp =  nleftemp.map(x => (ceil(x._1/2.0).toInt-> x._2))
                              .reduceByKey{ case (a,b) => { ((a._1 + b._1) ->  b._2)}}
        }
        
        val nleft:RDD[(Int, Int)] = nleftemp.map{case (x,y) => {(y._2 -> y._1)}}
        
        //compute nright = sum(Ncopies((N/2+1):N)>0)
        //filter to the correct keys
        val nright1 = nco.map(x => {val zeroval = x._1 % adj; 
                                    if (zeroval >= (adj/2 + 1) || zeroval == 0) (x._1 -> x._2) 
                                    else (x._1 -> -1)})
                         .filter( x => x._2 != -1)
        
        //compute Ncopies((N/2+1):N)>0
        var nrightemp: RDD[(Int, Int)] = nright1.map( x => {  if (x._2 > 0) (x._1 -> 1) else (x._1 -> 0)})
        
        //compute sum(Ncopies((N/2+1):N)>0)
        for(i <- 1 to maxIter-level){
          nrightemp =  nrightemp.map(x => (ceil(x._1/2.0).toInt -> x._2))
                                .reduceByKey(_+_)
        }
        
        val nright:RDD[(Int, Int)] = nrightemp.map(x => (ceil(x._1/2.0).toInt -> x._2) )
        
        //compute nlrtemp = nleft -nright
        val nlrtemp: RDD[(Int, Int)] = nleft.union(nright).coalesce(ztmprdd.partitions.size)
                                        .reduceByKey( (x,y) => abs(x - y) )
        
        //create nlr vector                                         
        val nlr: RDD[(Int, Int)] = nlrtemp.map(x => for(i <- x._1*adj to (x._1*adj-adj+1) by -1) yield (i -> x._2))
                                          .flatMap(x => x)
        
        //add nlr to osa : key | oldsample | nodeID | nlr   
        val osanlr: RDD[(Int, (Float, Int))] = osa.join(nlr)                 
        
        //add nlr to nco : key | ncopies | nodeID | nlr
        val nconlr: RDD[(Int, (Int, Int))] = nco.join(nlr)     
        
        //compute index = mod((nleft-nright)+(1:N/2)-1,N/2)+1
        //compute oldsamples2 = oldsamples(index)
        
        //filter the keys
        //key | oldsamples | nlr
        val oldsamples2t:RDD[(Int, (Float, Int))] = osanlr.filter( x => {
                                                               val validkey = x._1 % adj; validkey >= 1 && validkey <= (adj/2)})
        
        //compute and set the new key. store the old key in the value part :
        // newkey
        val oldsamples2newkey:RDD[(Int, (Int, Float))] = oldsamples2t.map{ case (x,y) => {val nodeID:Int = ceil(x/(Nlength/pow(2,level))).toInt;  
                                                                            val nkey = ((y._2 + x - 1) % (adj/2) + 1) + (nodeID-1)*adj;
                                                                            (nkey -> ( x -> Float.MinValue  ))}}
        
        // oldkey
        val oldsamples2oldkey:RDD[(Int, (Int, Float))] = oldsamples2t.map{case (x,y) => {(x -> ( Int.MinValue -> y._1))}}
        
        val oldsamples2temp:RDD[(Int, (Int, Float))] = (oldsamples2newkey.union(oldsamples2oldkey)).coalesce(ztmprdd.partitions.size)
                                                                          .reduceByKey( (x,y) => {
                                                                              val origkey = max(x._1,y._1);
                                                                              val origoldsamples = max(x._2,y._2);
                                                                              (origkey -> origoldsamples)})
        
        //oldsamples2 = oldsamples(index)                                                                              
        val oldsamples2:RDD[(Int, Float)] = oldsamples2temp.map{case (x,y)  => {(y._1 -> y._2)}}
        
        //compute f = cN(2:(N/2+1))>N/2
        //filter to the correct keys
        val ftemp:RDD[(Int, Int)] = cN.map( x => {val zeroval = x._1 % (adj+1);
                                            if (zeroval <= (adj/2 + 1) && zeroval >= 2 )  (x._1 -> x._2)
                                            else (x._1 -> -1)})
                                      .filter( x => x._2 != -1)
        
        //compute cN(2:(N/2+1))>N/2
        //oldkey
        val foldkey:RDD[(Int, (Int, Int))] = ftemp.map( d => {val nodeID:Int = ceil((d._1/(adj+1))).toInt; val key:Int = d._1;
                                                              if (d._2 > adj/2) (key ->  ( -1 -> 1))
                                                              else (key -> (-1 -> 0))})                                       
        
        //compute f=f(index)
        //newkey
        val fnewkey:RDD[(Int, (Int, Int))] = oldsamples2newkey.map{case (x,y) => {
                                                                   val nodeID:Int = ceil(x/(Nlength/pow(2,level))).toInt;
                                                                   val fv:Int = y._2.toInt;  //cast Float to int
                                                                   (x+nodeID -> ((y._1.toInt+nodeID) -> fv )) }}
        
        val fnc:RDD[(Int, Int)] =  (fnewkey.union(foldkey))
                                           .reduceByKey( (x,y) => {
                                               val origkey = max(x._1,y._1);
                                               val origoldsamples = max(x._2,y._2);
                                               (origkey -> origoldsamples)})  
                                           .map{case (x,y)  => {
                                               val nodeID:Int = ceil((x/(adj+1).toFloat)).toInt; //compute nodeID
                                               (y._1.toInt -> y._2)}}
        
        val fos:RDD[(Int, Float)] = fnc.map(x => x._1 -> x._2.toFloat)
        
        //compute olds2 = f.*oldsamples2
        val oldsamples2_old2:RDD[(Int, Float)] = oldsamples2.map(x => {val nodeID:Int = ceil(x._1/(Nlength/pow(2,level))).toInt;
                                                                  (x._1+nodeID -> x._2)})
        
        val olds2 = fos.union(oldsamples2_old2).reduceByKey(_*_) 
        
        //compute olds1 = ~f.*oldsamples(:,(N/2+1):N)
        //filter the correct olsamples values :  oldsamples(:,(N/2+1):N)
        //vecPar : key | ncopies | oldsamples | nodeID  ---> dat
        
        //osa --->  key | oldsamples | nlr
        val oldsa = osa.map(x => {val zeroval = x._1 % adj; 
                                  val nk:Int = adj/2;
                                  val nodeID:Int = ceil(x._1/(Nlength/pow(2,level))).toInt;
                                  if (zeroval >= (adj/2 + 1) || zeroval == 0) (x._1+nodeID-nk ->  x._2) 
                                  else (x._1+nodeID-nk -> Float.MinValue)})
                       .filter( x => x._2 != Float.MinValue)
        
        //not f : ~f
        //val notf = f.map( x => { if (x._2.toFloat == 0.0)  (x._1+adj/2,1.toString) else (x._1+adj/2,0.toString) })
        
        val notfnc:RDD[(Int, Int)]    = fnc.map( x => { if (x._2 == 0)  (x._1 -> 1) else (x._1 -> 0) })
        val notfos:RDD[(Int, Float)] = fos.map( x => { if (x._2 == 0)  (x._1 -> 1.0f) else (x._1 -> 0.0f) })
        
        val olds1:RDD[(Int, Float)] = (notfos.union(oldsa))
                                              .reduceByKey(_*_)
        
        //compute oldsamples_second = ~f.*oldsamples(:,(N/2+1):N) + f.*oldsamples2
        val oldsamples_second = (olds1.union(olds2)).map{ case (x,y) => {val nodeID:Int = ceil(x/(adj+1)).toInt;
                                                                     (x+adj/2-1-(nodeID-1)-1 -> y)}}
                                      .reduceByKey(_+_)              
        
        //compute N1 = ~f.*Ncopies((N/2+1):N)
        
        //filter to the correct keys for  : Ncopies((N/2+1):N)
        //vecPar : key | ncopies | oldsamples | nodeID  ---> dat                                         
        val N1c:RDD[(Int, Int)] = nco.map(x => {val zeroval = x._1 % adj;
                                      val nk:Int = adj/2;
                                      val nodeID:Int = ceil(x._1/(Nlength/pow(2,level))).toInt;
                                      if (zeroval >= (adj/2 + 1) || zeroval == 0) (x._1+nodeID-nk, x._2) 
                                      else (x._1+nodeID-nk -> -1)})
                           .filter( x => x._2 != -1)
        
        val N1 = (N1c.union(notfnc))
                                    .reduceByKey(_*_)
        
        //compute Nraw = Nraw(index)
        //Nrawoldkey
        val Nrawoldkey: RDD[(Int, (Int, Int))] = Nraw1.map(x => { (x._1 -> (-1 -> x._2 ))})          
        
        val Nrawnewkey: RDD[(Int, (Int, Int))] = oldsamples2newkey.map{case (x,y) => {(x -> (y._1 -> -1) )}}
        
        val Nraw2:RDD[(Int, (Int, Int))] =  (Nrawoldkey.union(Nrawnewkey))
                                                       .reduceByKey( (x,y) => { val origkey = max(x._1,y._1);
                                                                     val origoldsamples = max(x._2,y._2);
                                                                     (origkey -> origoldsamples)})
        
        val Nraw:RDD[(Int, Int)] = Nraw2.map{case (x,y)  => {val nodeID:Int = ceil(x/(Nlength/pow(2,level))).toInt;
                                            ((y._1+nodeID) -> y._2)}}
        
        //compute N2 = f.*Nraw(index)
        val N2:RDD[(Int, Int)] = (Nraw.union(fnc))
                                      .reduceByKey(_*_)
        
        //compute Nsecond = ~f.*Ncopies((N/2+1):N) + f.*Nraw(index)
        val Nsecond:RDD[(Int, Int)] = (N1.union(N2))
                                         .map(x  => {val nodeID:Int = ceil(x._1/(adj+1)).toInt;
                                                     (x._1+adj/2-1-(nodeID-1)-1 -> x._2)})
                                         .reduceByKey(_+_)
	
        nco = Nfirst.union(Nsecond).repartition(ztmprdd.partitions.size)
	osa =  oldsamples_first.union(oldsamples_second).repartition(ztmprdd.partitions.size)
        nco.count
	osa.count
    }
    
    osa
  }
  
  def cumulativeSummationred(in:RDD[(Int,Int)], NumIter:Int) : RDD[(Int,Int)] = {
    
    val iter:Int = 1
    val storeintlists: ListBuffer[RDD[(Int, Int)]] = sumcsmred(in, NumIter, iter, ListBuffer(in))
    
    //initial parent node; root of the balanced binary tree
    val root:RDD[(Int, Int)] = storeintlists(NumIter)
    
    val opiter:Int = NumIter
    val output:RDD[(Int,Int)] = backstepsred(root, NumIter, opiter, storeintlists)
    
    output
  }
  
  def backstepsred(in:RDD[(Int,Int)], NumSteps:Int, opit:Int, store:ListBuffer[RDD[(Int, Int)]]) : RDD[(Int,Int)]  = { 
    val t2:RDD[(Int, Int)] = store(opit-1)
    
    //prepare key for the right child
    val rchild = in.map(x => (x._1*2 -> x._2))
    
    //join level i-1 with right children
    val out1 = t2.join(rchild)
    
    //duplicate the parent nodes
    val out2 = out1.map(x => for(i <- x._1 to x._1-1 by -1) yield (i -> x._2)  ) 
                   .flatMap(x => x)
    
    //create the parent nodes for the next level               
    val out3 = out2.map{case (x,(y,z)) => if (x % 2 == 0) (x,z) else (x,z-y) }
    
    if (opit == 1) (out3) else backstepsred((out3), NumSteps, opit-1, store)
  }
  
  def sumcsmred(in:RDD[(Int,Int)], NumSteps:Int, it:Int, store:ListBuffer[RDD[(Int, Int)]]) : ListBuffer[RDD[(Int, Int)]]  = { 
     val list =  in.map(x => (ceil(x._1/2.0).toInt -> x._2))
              .reduceByKey(_+_)
     store ++= ListBuffer(list)
     if (it == NumSteps) store else sumcsmred(list, NumSteps, it+1, store)
  }
  
  /****************
   *  Bitonic Sort
   ****************/
  
  //bitonic sort
  def bitonicSort(nc: RDD[(Int,Int)], os: RDD[(Int,Float)], NumIter: Int, Nlength: Int) : RDD[(Int, (Int, Float))] = {
    //initialise sequence for bitonic sort
    //var blist = nc.map(x => x._1 -> x._2.toString).union(os.map(x => x._1 -> x._2.toString)).coalesce(1)
    //              .reduceByKey((x,y) => x + "," + y)
    
    var blist = nc.join(os)              
    
    /*******************************************
     *  Phase 1 : Create the bitonic sequence  *
     *******************************************/
    
    //initialisation
    var va = Nlength/2;
    var chunk = 0
    var offset = 0;
    
    //total number of main steps
    val Nsteps = (NumIter - 1).toInt
    
    for(steps <- 1 to Nsteps){
        for(iter <- steps to 1 by -1){
            
            blist = blist.map({case  (x, y) => mapperA(x, iter, steps, Nlength, va, chunk, y)}) 
                                   .reduceByKey((x,y) => x.union(y))
                                   .flatMap({case (x,y) => mapperB(x,y)})
            
            chunk = va
            offset = va;
            va = va/2
        }
    }
    
    /*****************************************
     *  Phase 2 : Sort the bitonic sequence  *
     *****************************************/
    
    val sNumSteps:Int = NumIter.toInt
    val maxIter:Int = (log(Nlength)/log(2)).toInt
    
    //step 1
    blist = blist.map({case  (x, y) => mapperSBS1(x, Nlength, y)}) 
                    .reduceByKey((x,y) => x.union(y))
                    .flatMap({case (x,y) => mapperB(x,y)})
    
    //step 2
    blist = blist.map({case  (x, y) => mapperSBS2(x, Nlength, y)}) 
                    .reduceByKey((x,y) => x.union(y))
                    .flatMap({case (x,y) => mapperB(x,y)})
    
    //remaining steps
    for(i <- 3 to sNumSteps){
        blist = blist.map({case  (x, y) => mapperSBS(x, i, Nlength, y)}) 
                        .reduceByKey((x,y) => x.union(y))
                        .flatMap({case (x,y) => mapperB(x,y)})
    }
    
    /**********
     * Return *
     **********/
    
    //return output in correct form 
    (blist)
    
  }

  
  def redmerge( x : Int, y : String) : List[(Int,String)] = {
      var newlist = List[(Int,String)]()
      newlist
  }
  
  def mapperSBS2(akey: Int, le: Int, tval: (Int,Float)) : (Int, ArrayBuffer[(((Int, Float), Int), Int)]) = {
    var length:Int = le
    var offset:Int = length/2
    var chunk:Int = length/2
    var va:Int = length/4
    var sep:Int = -1
    
    var add:Int = 0
    
    var k:Int = 0
    
    if (akey <= chunk){
        k = akey % va
        
        if(akey > chunk/2){
            sep = 1
        }else{
            sep = 0
        }
    }else{
        k = akey % va + offset/2
        
        if(akey > chunk/2 + offset){
            sep = 1
        }else{
            sep = 0
        }
    }

    
    //"value, sep, oldkey"
    var outputValue =  tval._1 -> tval._2 -> sep -> akey
    
    //new key, "value, sep, oldkey"
    (k -> ArrayBuffer(outputValue))
  }
  
  def mapperSBS1(akey: Int, le: Int, tval: (Int,Float)) : (Int, ArrayBuffer[(((Int, Float), Int), Int)]) = {
    var length:Int = le
    var va:Int = le/2
    var sep:Int = -1
    var add:Int = 0
    
    var k:Int = akey % va
    
    if(akey > length/2){
        sep = 1
    }else{
        sep = 0
    }
    
    //"value, sep, oldkey"
    var outputValue = tval._1 -> tval._2 -> sep -> akey
    
    //new key, "value, sep, oldkey"
    (k -> ArrayBuffer(outputValue))
  }
  
  def mapperSBS(akey: Int, MR: Int, le: Int, tval: (Int,Float)):(Int, ArrayBuffer[(((Int, Float), Int), Int)]) = {
    var MRstep:Int = MR
    var length:Int = le
    var va:Int = (le/(pow(2,(MR)))).toInt
    
    var chunk:Int = (le/(pow(2,(MR-1)))).toInt
    var offset:Int = (le/(pow(2,(MR-1)))).toInt
    
    var k:Int = 0
    var sep:Int = -1
    
    var add:Int = 0
    
    var maxIter:Int = (log(length)/log(2.0).toDouble).toInt
    
    if (MRstep == maxIter){
        k = ceil(akey/2.0).toInt
        
        if(akey % 2 == 0){
            sep = 1
        }else{
            sep = 0
        }
    }else{
        if(akey <= chunk){
            k = akey % va
            
            if (akey > chunk/2){
                sep = 1
            }else{
                sep = 0
            }
        }else{
            add = ((ceil(akey/chunk.toDouble) - 1) * chunk/2 ).toInt
            k = akey % va + add
            
            if(akey > chunk/2 + offset * ceil(akey/chunk) || akey % chunk == 0){
                sep = 1
            }else{
                sep = 0
            }
        }
    }
    
    //"value, sep, oldkey"
    var outputValue = tval._1 -> tval._2 -> sep -> akey
    
    //new key, "value, sep, oldkey"
    (k -> ArrayBuffer(outputValue))
  }
  
  def mapperA(akey: Int, MR: Int, s: Int, le: Int, v: Int, ch: Int, tval: (Int,Float)):(Int, ArrayBuffer[(((Int, Float), Int), Int)]) = {
    //new key
    var k:Int = 0
    var sep:Int = -1
    var add = 0
    
    var MRstep:Int = MR
    var step:Int = s
    var length:Int = le
    var va:Int = v
    var chunk:Int = ch
    
    //ncopies, oldsamples
    
    var side = 0
    var rside = 0
    
    if (MRstep == 1){
        va = length/2
        k = ceil(akey.toDouble/2.0).toInt
        
        var c:Int = floor(akey.toDouble/2.0).toInt % 2
        
        if(c == 1){
          sep = 1
        }else{
          sep = 0
        }
        
        if (step > 1){
            side = ceil(akey.toDouble/pow(2,step).toDouble).toInt
            
            if(side % 2 == 1){
                if(akey % 2 == 1){
                  sep = 0
                }else{
                  sep = 1
                }
            }else{
                if(akey % 2 == 1){
                  sep = 1
                }else{
                  sep = 0
                }
            }
        }
    }else if(MRstep == 2){
        chunk = 4
        va = 4/2
        
        if(akey <= chunk){
            k = akey % va
            
            if(akey > chunk.toDouble/2.0){
                sep = 1
            }else{
                sep = 0
            }
        }else{
            add = ((ceil(akey.toDouble/chunk.toDouble)-1.0)*chunk/2.0).toInt
            
            k =  akey % va + add
            
            side = ceil(akey.toDouble/pow(2,step).toDouble).toInt
            rside = ceil(akey.toDouble/2.0).toInt
            
            if(side % 2 ==1){
                if(rside % 2 == 1){
                    sep = 0
                }else{
                    sep = 1
                }
            }else{
                if(rside % 2 == 1){
                    sep = 1
                }else{
                    sep = 0
                }
            }
        }
    }else{
        va = pow(2,MRstep).toInt
        chunk = va
        va = va/2
        
        if(akey <= chunk){
            //generate new key
            k = akey % va
            
            if(akey > chunk/2){
                sep = 1
            }else{
                sep = 0
            }
            
        }else{
            add = ((ceil(akey/chunk.toDouble) - 1.0)*chunk/2.0).toInt
            k = akey % va + add
            
            side = ceil(akey.toDouble/pow(2.0,step).toDouble).toInt
            rside = ceil(akey.toDouble/va.toDouble).toInt
            
            if(side % 2 == 1){
                if(rside % 2 == 1){
                    sep = 0
                }else{
                    sep = 1
                }
            }else{
                if(rside % 2 == 1){
                    sep = 1
                }else{
                    sep = 0
                }
            }
        }
    }
    
    //"value, sep, oldkey"
    val outputValue = tval._1 -> tval._2 -> sep -> akey
    
    //new key, "value, sep, oldkey"
    (k -> ArrayBuffer(outputValue))
  }
  
  def mapperB(a:Int, b: ArrayBuffer[(((Int, Float), Int), Int)]):ArrayBuffer[(Int, (Int, Float))] = {
    val outV = b  
    
    val mlist = new ArrayBuffer[(Int, (Int, Float))]()
    
    if(outV(0)._1._2 == 1){
      if (outV(0)._1._1._1 > outV(1)._1._1._1){
          mlist .+= (( outV(1)._2,  (outV(0)._1._1._1, outV(0)._1._1._2 ) ))
          mlist .+= (( outV(0)._2, ( outV(1)._1._1._1, outV(1)._1._1._2 ) ))
      }else{
          mlist .+= (( outV(0)._2,  (outV(0)._1._1._1, outV(0)._1._1._2 ) ))
          mlist .+= (( outV(1)._2, ( outV(1)._1._1._1, outV(1)._1._1._2 ) ))
      }
    }else{
      if(outV(0)._1._1._1 < outV(1)._1._1._1){
          mlist .+= (( outV(1)._2,  (outV(0)._1._1._1, outV(0)._1._1._2 ) ))
          mlist .+= (( outV(0)._2, ( outV(1)._1._1._1, outV(1)._1._1._2 ) ))
      }else{
          mlist .+= (( outV(0)._2,  (outV(0)._1._1._1, outV(0)._1._1._2 ) ))
          mlist .+= (( outV(1)._2, ( outV(1)._1._1._1, outV(1)._1._1._2 ) ))
      }
    }
    
    (mlist)
  }
 
  
  /*******************************
   *  Minimum Variance Resampling
   *******************************/
  
  //minimum variance resampling
  def minvarresample(lweights: RDD[(Int,Float)], NumIter: Int, Nlength: Int, sc: SparkContext) : RDD[(Int,Int)] = {
      //compute the cumsum of the list
      val csmw: RDD[(Int, Float)] = cumulativeSummationmvr(lweights, NumIter)
      
      //broadcast random value
      val randNum = sc.broadcast(Random.nextFloat)
      
      //compute w = floor(w*N+rand)
      val csmwf: RDD[(Int, Int)] = csmw.mapValues(a => (floor(a * Nlength + 0.5f * randNum.value)).toInt)
      
      //odd values of difference between adjacent elements
      val w1: RDD[(Int, Int)] = csmwf.map(x  => (floor(x._1/2.0).toInt -> x._2))
                                    .filter(x => x._1 < Nlength/2)
                                    .reduceByKey((x,y) =>  math.abs(x-y))
                                    .map(x => if (x._1 == 0) (x._1+1 -> x._2.toInt) else (2*x._1+1 -> x._2.toInt))
      
      //even values of difference between adjacent elements
      val w2: RDD[(Int, Int)] = csmwf.map(x => (ceil(x._1/2.0).toInt -> x._2))
                                      .filter(x => x._1 > 0)
                                      .reduceByKey{ (x,y) => { math.abs(x-y)}}
                                      .map(x => (2*x._1 -> x._2.toInt))
      
      //merge to have the Ncopies
      val Ncopiesrdd = w1.union(w2).coalesce(csmwf.partitions.size)
      
      (Ncopiesrdd)
  }
  
  def cumulativeSummationmvr(in:RDD[(Int,Float)], NumIter:Int) : RDD[(Int,Float)] = {
    
    val iter:Int = 1
    val storeintlists: ListBuffer[RDD[(Int, Float)]] = sumcsmmvr(in, NumIter, iter, ListBuffer(in))
    
    //initial parent node; root of the balanced binary tree
    val root:RDD[(Int, Float)] = storeintlists(NumIter)
    
    val opiter:Int = NumIter
    val output:RDD[(Int,Float)] = backstepsmvr(root, NumIter, opiter, storeintlists)
    
    output
  }
  
  def sumcsmmvr(in:RDD[(Int,Float)], NumSteps:Int, it:Int, store:ListBuffer[RDD[(Int, Float)]]) : ListBuffer[RDD[(Int, Float)]]  = { 
     val list =  in.map(x => (ceil(x._1/2.0).toInt -> x._2))
              .reduceByKey(_+_)
     store ++= ListBuffer(list)
     if (it == NumSteps) store else sumcsmmvr(list, NumSteps, it+1, store)
  }

  def backstepsmvr(in:RDD[(Int,Float)], NumSteps:Int, opit:Int, store:ListBuffer[RDD[(Int, Float)]]) : RDD[(Int,Float)]  = { 
    val t2:RDD[(Int, Float)] = store(opit-1)
    
    //prepare key for the right child
    val rchild = in.map(x => (x._1*2 -> x._2))
    
    //join level i-1 with right children
    val out1 = t2.join(rchild)
    
    //duplicate the parent nodes
    val out2 = out1.map(x => for(i <- x._1 to x._1-1 by -1) yield (i -> x._2)  ) 
                   .flatMap(x => x)
    
    //create the parent nodes for the next level               
    val out3 = out2.map{case (x,(y,z)) => if (x % 2 == 0) (x,z) else (x,z-y) }
    
    if (opit == 1) (out3) else backstepsmvr((out3), NumSteps, opit-1, store)
  }
  
  def mapOddDiff(xx: Int,b: Float):(Int, Float) = {
      var a  = floor(xx/2.0).toInt
      (a, b)
  }
  
  def mapEvenDiff(xx: Int,b: Float):(Int, Float) = {
      var a  = ceil(xx/2.0).toInt
      (a, b)
  }
  
  /*************************
   *  Effective Sample Size
   *************************/
  
  //Compute the Effective Sample Size, Neff
  def EffectiveSampleSize(lweights: RDD[(Int,Float)], NumIter: Int) : Float = {
      
      //compute sum(exp(vec_lweights).^2)
      var sumExpVecl_weightsrdd = lweights.mapValues(x => exp(x)*exp(x))
      
      for(iterEff <- 1 to NumIter){
        sumExpVecl_weightsrdd = sumExpVecl_weightsrdd.map(x => (ceil(x._1/2.0).toInt -> x._2))
                                                     .reduceByKey(_+_)
      }
      
      //retrieve the total sum pair of the sum(exp(vec_lweights)) and store the value part of the pair
      val sumExpWeightNeffValue = sumExpVecl_weightsrdd.first._2.toFloat
      
      //compute effective sample size
      val Neff:Float = 1.0f/sumExpWeightNeffValue
      
      Neff
  }
  
  /*************************
   *  Weights Normalization
   *************************/
  
  //Compute the Normalized Weights
  def WeightsNormalization(lweights: RDD[(Int,Float)], NumIter: Int, Nlength:Int) : RDD[(Int,Float)] = {
      //compute max value of particle weights
      var max_vecl_weights = lweights
      
      for(iter1 <- 1 to NumIter){
        max_vecl_weights =  max_vecl_weights.map(x => (ceil(x._1/2.0f).toInt -> x._2))
                                            .reduceByKey( (a,b) => if (a >= b) a else b )  //max value
      }
      
      //retrieve the pair with the max value of the lweights and create a vector with size Nlength
      //containing the max value for all keys
      var maxlWeightpair = max_vecl_weights.map(x => for(i <- Nlength to 1 by -1) yield (i -> x._2)  ) 
                                           .flatMap(x => x)
      
      //compute vec_lweights = vec_lweights - max(vec_lweights)
      val vecl_weightsrdd = lweights.join(maxlWeightpair).mapValues(x => x._1 - x._2)                                       
      
      //compute sum(exp(vec_lweights))
      var sumExpVecl_weightsrdd = vecl_weightsrdd.mapValues(x =>  exp(x).toFloat)
      
      for(iter2 <- 1 to NumIter){
        sumExpVecl_weightsrdd = sumExpVecl_weightsrdd.map(x => (ceil(x._1/2.0).toInt -> x._2))
                                                     .reduceByKey(_+_)
      }
      
      //create a vector with size Nlength containing in all the values the log(sum(exp(vec_lweights)))
      var sumExpWeigths = sumExpVecl_weightsrdd.map(x => for(i <- Nlength to 1 by -1) yield (i -> log(x._2).toFloat)  ) 
                                               .flatMap(x => x)
      
      //compute vec_lweights = vec_lweights - log(sum(exp(vec_lweights)))
      val output = vecl_weightsrdd.join(sumExpWeigths).mapValues(x => x._1 - x._2)  
      
      output
  }
  
  def randomNumberAt(index: Int, globalSeed:Int) : Float = ((new Random(globalSeed + index)).nextGaussian).toFloat
  
}
