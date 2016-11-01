/***********************************************************
*                 Particle filter                          *
* Details : https://github.com/particlefilter/mrpf/        *
* Copyright(c)2016 University of Liverpool                 *
************************************************************/

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.ProcessBuilder.Redirect;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class PF_Hadoop extends Configured implements Tool {
	
    public static void main(String[] args) throws Exception {
        int res = ToolRunner
                .run(new Configuration(), new PF_Hadoop(), args);
        System.exit(res);
    }
	
	
    @Override
    public int run(String[] args) throws Exception {
    	
        //number of rows in file
        int Nlength = Integer.parseInt(args[2]);
        
        //input size must be power of 2
        if ((Nlength & (Nlength-1)) != 0){
        	System.out.println("Error: The number of elements is not a power of 2.");
        	return 0;
        }
        
        /**Initialization*/
        
        int NumIter = (int) (Math.log(Nlength)/Math.log(2));
        
        float Qk_1 = 10.0f; // variance of the system noise
        float Rk = 1.0f;    // variance of the measurement noise
        int time = 1;       // number of iterations
        boolean flag = true; 
        
        //Update the system and measurement equations
        float x = 1.0f; // initial actual state
        float initVar = 10.0f; //variance of the initial estimate
        
        /**STEP 1: CREATE PARTICLES*/
		Configuration conf0red = new Configuration();
		conf0red.set("x", Float.toString(x));
		conf0red.set("initVar", Float.toString(initVar));
		
    	Job job0 = Job.getInstance(conf0red, "Job0");
    	job0.setJarByClass(PF_Hadoop.class);
    	job0.setOutputKeyClass(Text.class);
    	job0.setOutputValueClass(Text.class);
    	
    	job0.setMapperClass(Map_createparticles.class);
    	job0.setNumReduceTasks(0);
    	
    	job0.setInputFormatClass(TextInputFormat.class);
    	job0.setOutputFormatClass(TextOutputFormat.class);
    	
    	String input = args[0];
    	String output = "vec_Particles";
    	
    	TextInputFormat.addInputPath(job0, new Path(input));
    	TextOutputFormat.setOutputPath(job0, new Path(output));
        
    	job0.waitForCompletion(true);
        
    	
    	Random r = new Random();
        float z = 1.0f;
        float randn = 1.0f;
        float xestimate = 0.0f;
        
        //data for plotting
        float[] xreal = new float[time];
        float[] xest = new float [time];
        double[] etime = new double [time];
        
        
        //Initialize Configuration
    	Configuration[] confList = new Configuration[NumIter];
    	
    	for(int i=0;i<NumIter;i++){
    		confList[i] = new Configuration();
    		confList[i].set("iter", Float.toString(i+1));
    		confList[i].set("Size", Float.toString(Nlength));
    	}
    	
        
        //Initialize configuration
    	Configuration[][] confListred = new Configuration[NumIter][NumIter];
	    
    	int maxIter = NumIter;
    	
    	for(int j=0;j<maxIter;j++){
            int iter = j;
            int adj = (int) (Nlength/Math.pow(2,iter));
            int NumIterc = (int)(Math.log(adj)/Math.log(2));
            
            int hcNend = (int) (adj/2) + 1;
            int adjp1 = adj+1;
            
        	int halfN = hcNend-1;
        	
	        for(int i=0;i<NumIter;i++){
	        	confListred[j][i] = new Configuration();
	        	//conf2List[i] = this.getConf();
	        	confListred[j][i].set("iter", Integer.toString(i));
	        	confListred[j][i].set("length", Integer.toString(Nlength));
	        	
	        	confListred[j][i].set("Size", Float.toString((float) NumIterc));
	    		confListred[j][i].set("adj", Integer.toString(adj));
	    		
	    		confListred[j][i].set("hcNend", Integer.toString(hcNend));
	    		confListred[j][i].set("adjp1", Integer.toString(adjp1));
	    		
	    		confListred[j][i].set("halfN", Integer.toString(halfN));
	        }
    	}
	    
    	//Start the particle filter
    	for(int t=1;t<=time;t++){
    		
    		//elapsed time
    		long startTime = System.nanoTime();
    		
    		//random number
            randn = (float) r.nextGaussian();
            
    		/**Update the system and measurement equation*/
            //system equation and system noise
    		x = (float) (0.5*x + 25.0*x/(1.0+x*x) + 8.0*Math.cos(1.2*(t-1)) + Math.sqrt(Qk_1)*randn);
    		//measurement equation and measurement noise
    		randn = (float) r.nextGaussian();
    		z = (float) (Math.pow(x,2)/20.0 + Math.sqrt(Rk)*randn);
    		
    		/**New State*/
			
    		Configuration conf1red = new Configuration();
    		conf1red.set("Qk_1", Float.toString(Qk_1));
    		conf1red.set("t", Float.toString((float)t));
    		
        	Job job1 = Job.getInstance(conf1red, "Job1");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map_newstate.class);
        	job1.setNumReduceTasks(0);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	input = "vec_Particles";
        	output = "new_state";
        	
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
    		
    		/**Measurement Update and new Particle Weights*/
    		
        	Configuration conf2red = new Configuration();
    		conf2red.set("Rk", Float.toString(Rk));
    		conf2red.set("z", Float.toString(z));
    		
        	Job job2 = Job.getInstance(conf2red, "Job2");
        	job2.setJarByClass(PF_Hadoop.class);
        	job2.setOutputKeyClass(Text.class);
        	job2.setOutputValueClass(Text.class);
        	
        	job2.setMapperClass(Map_neweights.class);
        	job2.setNumReduceTasks(0);
        	
        	job2.setInputFormatClass(TextInputFormat.class);
        	job2.setOutputFormatClass(TextOutputFormat.class);
        	
        	input = "new_state/part-m-00000";
        	output = "vec_Weights";
        	
        	TextInputFormat.addInputPath(job2, new Path(input));
        	TextOutputFormat.setOutputPath(job2, new Path(output));
            
        	job2.waitForCompletion(true);
        	
    		/**Weights Normalization */
    		weightsNormalization(confList);
    		
    		/**Effective Sample Size*/
    		float Neff = effsamplesize(confList);
    		
    		//threshold
        	float Nt = Nlength*0.50f;
        	
        	
        	if (Neff <= Nt){
        		/**STEP 3: RESAMPLING*/
        		
        		/**Minimum Variance Resampling */
        		mvr(Nlength, confList);
        		
        		/**Bitonic Sort */
        		bitonicSort(NumIter, Nlength);
        		
        		/**Execute the Redistribution */
        		redistribute2(NumIter, Nlength, confListred);
        		
        		/** Estimate the mean value of the posterior distribution*/
        		
        		flag = true;
        		xestimate = xestm(Nlength, flag, confList);
        		
        	}else{
            	FileSystem fs = FileSystem.get(confList[0]);
            	FileStatus[] fileStatus = fs.listStatus(new Path("."));
            	Path[] paths = FileUtil.stat2Paths(fileStatus);
            	for (Path path : paths){
            		if (!(path.getName().equals("src")) && !(path.getName().equals("bin")) && !(path.getName().equals(".settings")) && !(path.getName().equals("weights_output")) && !(path.getName().equals("input0")) && !(path.getName().equals("new_state")) && !(path.getName().equals(".classpath")) && !(path.getName().equals(".project"))){
            			fs.delete(path, true);
            		}
            	}
            	
        		fs.rename(new Path("new_state"), new Path("vec_Particles"));
        		
        		/** Estimate the mean value of the posterior distribution*/
        		
        		flag = false;
        		xestimate = xestm(Nlength, flag, confList);
        	}
        	
    		//elapsed time
        	long elapsedTime = System.nanoTime() - startTime;
        	double ctime = (double)elapsedTime/1000000000.0;
    		
    		//data for plotting
    		xreal[t-1] = x;
    		xest[t-1] = xestimate;
    		etime[t-1] = ctime; 
    		
    	}//end_for_loop_time
        
        //print results
    	System.out.println("========================");
    	System.out.println("Number of steps = " + time);
        for (int i=0;i<time;i++){
        	System.out.println("Real value \t= "+ xreal[i]);
        	System.out.println("SIR estimation \t= "+ xest[i]);
        	System.out.println("Total time \t= "+ etime[i]);
        	System.out.println("---");
        }
        System.out.println("========================");
        
		return 0;
    }
	
	
    /********************
     * Create Particles *
     * ******************/
	
	//Job 4
    public static class Map_createparticles extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String xtemp = context.getConfiguration().get("x");
            String initVartemp = context.getConfiguration().get("initVar");
            
            //Generate Guassian Random Numbers
            Random r = new Random();
            float randn = (float) r.nextGaussian();
            
            float x = Float.parseFloat(xtemp);
            float initVar = Float.parseFloat(initVartemp);
            float vpart = (float) (x + Math.sqrt(initVar)+ randn);
            
            outputValue.set(indicesAndValue[0] +","+ Float.toString(vpart));
            context.write(null, outputValue);
        }
    }
    
    
    /*************
     * New State *
     * ***********/
	
	//Job 4
    public static class Map_newstate extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String Qk_1temp = context.getConfiguration().get("Qk_1");
            String ttemp = context.getConfiguration().get("t");
            
            //Generate Guassian Random Numbers
            Random r = new Random();
            float randn = (float) r.nextGaussian();
            
            float Qk_1 = Float.parseFloat(Qk_1temp);
            float t = Float.parseFloat(ttemp);
            
            float vec_Particles = Float.parseFloat(indicesAndValue[1]);
            float vpart = (float) (0.5*vec_Particles + 25.0*vec_Particles/(1.0+vec_Particles*vec_Particles) + 8.0*Math.cos(1.2*(t-1)) + Math.sqrt(Qk_1)*randn);
            
            outputValue.set(indicesAndValue[0] +","+ Float.toString(vpart));
            context.write(null, outputValue);
        }
    }
    
    /**********************************
     * New Measurement and Weights *
     * ***************************/
	
	//Job 4
    public static class Map_neweights extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String Rktemp = context.getConfiguration().get("Rk");
            String ztemp = context.getConfiguration().get("z");
            
            float Rk = Float.parseFloat(Rktemp);
            float z = Float.parseFloat(ztemp);
            
            float new_state = Float.parseFloat(indicesAndValue[1]);
            
            //measurement update
            float new_measur = (float) ((new_state*new_state)/20.0);
            
            float t1 = (float) Math.sqrt(2.0*Math.PI*Rk);
            float t2 = (float) ((z-new_measur)*(z-new_measur));
            
            float new_weights = (float) (-Math.log(t1)-(t2/(2.0*Rk)));
            //vec_Weights[i] = -Math.log(t1)-(t2/(2.0*Rk));
            
            outputValue.set(indicesAndValue[0] +","+ Float.toString(new_weights));
            context.write(null, outputValue);
        }
    }
    
	/*************************
	 * Weights Normalization *
	 *************************/
	
    //Job 1 - find max
    public static class Map1max_weig extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1max_weig extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float maxval = -Float.MAX_VALUE;
            
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            for (Text val : values) {
            	value = val.toString();
            	maxval = Math.max(Float.parseFloat(value), maxval);
            }
            
            if (Math.log(Float.parseFloat(Nlength))/Math.log(2) == Float.parseFloat(iter)){
            	int fkey = 2*Integer.parseInt(key.toString()); 
            	context.write(null, new Text(fkey+","+Float.toString(maxval)));
            }else{
            	context.write(null, new Text(key+","+Float.toString(maxval)));
            }
        }
    }
    
    // w = w - max(w)
    public static class Map2_weig extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String maxweights = context.getConfiguration().get("maxw");
            
            float maxw = Float.parseFloat(maxweights);
            
            float weights = Float.parseFloat(indicesAndValue[1]);
            weights = weights-maxw;
            
            outputValue.set(indicesAndValue[0] +","+ Float.toString(weights));
            context.write(null, outputValue);
        }
    }
    
    //find log(sum(exp(w)))
    public static class Map3_weig extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String iter = context.getConfiguration().get("iter");
            
            float iteration = Float.parseFloat(iter);
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            int k = (int) Math.ceil(kin/2.0);
            
            float val = Float.parseFloat(indicesAndValue[1]);
            
            if (iteration == 1.0f){
            	 val = (float) Math.exp((float) val);
            }
            
            outputKey.set(Integer.toString(k));
            outputValue.set(Float.toString(val));
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce3_weig extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float sum = 0.0f;
            
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            for (Text val : values) {
            	value = val.toString();
            	sum += Float.parseFloat(value);
            }
            
            if (Math.log(Float.parseFloat(Nlength))/Math.log(2) == Float.parseFloat(iter)){
            	int fkey = 2*Integer.parseInt(key.toString()); 
            	float lw = (float) Math.log(sum);
            	context.write(null, new Text(fkey+","+Float.toString(lw)));
            }else{
            	context.write(null, new Text(key+","+Float.toString(sum)));
            }
        }
    }
    
    public static class Map4_weig extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String lgsmexpw = context.getConfiguration().get("lgsmexpw");
            
            float lsew = Float.parseFloat(lgsmexpw);
            
            float weights = Float.parseFloat(indicesAndValue[1]);
            weights = weights-lsew;
            
            outputValue.set(indicesAndValue[0] +","+ Float.toString(weights));
            context.write(null, outputValue);
        }
    }
    
    /*************************
     * Effective Sample Size *
     *************************/
    
    //find sum(exp(w.^2))
    public static class Map1_Neff extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String iteration = context.getConfiguration().get("iter");
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            int k = (int) Math.ceil(kin/2.0);
            
            float val = Float.parseFloat(indicesAndValue[1]);
            
            if (Float.parseFloat(iteration) == 1.0f){
            	float temp = 0.0f;
            	temp = (float) (Math.exp(val)*Math.exp(val));
            	val = temp;
            }
            
            outputKey.set(Integer.toString(k));
            outputValue.set(Float.toString(val));
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1_Neff extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float sum = 0.0f;
            
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            for (Text val : values) {
            	value = val.toString();
            	sum += Float.parseFloat(value);
            }
            
            if (Math.log(Float.parseFloat(Nlength))/Math.log(2) == Float.parseFloat(iter)){
            	int fkey = 2*Integer.parseInt(key.toString()); 
            	context.write(null, new Text(fkey+","+Float.toString(sum)));
            }else{
            	context.write(null, new Text(key+","+Float.toString(sum)));
            }
        }
    }
    
	/*******************************
	 * Minimum Variance Resampling *
	 *******************************/
	
	//Job 1 - Cumsum phase 1 : summation
    public static class Map1 extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1 extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float summation = 0.0f;
            
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            for (Text val : values) {
            	value = val.toString();
            	summation += Float.parseFloat(value);
            }
            
            if (Math.log(Float.parseFloat(Nlength))/Math.log(2) == Float.parseFloat(iter)){
            	int fkey = 2*Integer.parseInt(key.toString()); 
            	context.write(null, new Text(fkey+","+Float.toString(summation)));
            }else{
            	context.write(null, new Text(key+","+Float.toString(summation)));
            }
        }
    }
    
	//Job 2 - Cumsum phase 2
    public static class Map2 extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(indicesAndValue[1]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce2 extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            List<Float> frvalues = new ArrayList<Float>();
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	frvalues.add(Float.parseFloat(value[0]));
            }
            
            int rchild = 2*Integer.parseInt(key.toString());
            int lchild = 2*Integer.parseInt(key.toString())-2;
            
            if (Integer.parseInt(key.toString()) % 2 == 0){
            	
            	if (Math.log(Float.parseFloat(Nlength))/Math.log(2) == Float.parseFloat(iter)){
            		context.write(null, new Text(lchild/2+ "," + Float.toString(Collections.min(frvalues)-Collections.max(frvalues))));
            		context.write(null, new Text(rchild/2+ "," + Float.toString(Collections.min(frvalues))));
            	}else{
            		context.write(null, new Text(lchild+ "," + Float.toString(Collections.min(frvalues)-Collections.max(frvalues))));
            		context.write(null, new Text(rchild+ "," + Float.toString(Collections.min(frvalues))));
            	}
            }
        }
    }
    
	//Job 3
    public static class Map3 extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String randNum = context.getConfiguration().get("randNum");
            String Nwend = context.getConfiguration().get("Nwend");
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            float nval = Float.parseFloat(indicesAndValue[1])*Float.parseFloat(Nwend) + Float.parseFloat(randNum);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(Integer.toString(kin)+","+Integer.toString((int)nval));
            
            context.write(null, outputValue);
        }
    }
    
	//Job 4 - Generate the odd values of the diff function
    public static class Map4 extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce4 extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            List<Integer> tvalues = new ArrayList<Integer>();
            
            for (Text val : values) {
            	value = val.toString();
            	
            	tvalues.add(Integer.parseInt(value));
            }
            
            //write the odd values of the final output
            if (tvalues.size() > 1){
            	int nkey = 2*Integer.parseInt(key.toString());
            	context.write(null, new Text(nkey + ","+ Integer.toString((Math.abs(tvalues.get(0)- tvalues.get(1))))));
            }
        }
    }
    
	//Job 5 - Generate the even values of the diff function
    public static class Map5 extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.floor(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce5 extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            List<Integer> tvalues = new ArrayList<Integer>();
            
            for (Text val : values) {
            	value = val.toString();
            	
            	tvalues.add(Integer.parseInt(value));
            }
            
            //write the 1st value of the final output
            if (Integer.parseInt(key.toString()) == 0){
            	int nkey = Integer.parseInt(key.toString())+1;
                context.write(null, new Text(nkey + ","+Integer.toString(tvalues.get(0))));
            }
            
            //write the even values of the final output
            if (tvalues.size() > 1){
                int nkey = 2*Integer.parseInt(key.toString())+1;
                context.write(null, new Text(nkey + ","+Integer.toString((Math.abs(tvalues.get(0)- tvalues.get(1))))));
            }
        }
    }
    
    /****************
     * Bitonic Sort *
     ****************/
    

    /************************************
     *   Construct the input for the BS *
     * **********************************/
	
    public static class MapNcopies_InputBS extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - value
            //indicesAndValue[2] - nodeID
            
            int v = Integer.parseInt(indicesAndValue[1]); //Ncopies
            float olds = Float.MIN_VALUE; //oldsamples
            
            outputKey.set(indicesAndValue[0]);
            outputValue.set(Integer.toString(v)+","+Float.toString(olds));
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class MapOldsamples_InputBS extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - value
            //indicesAndValue[2] - nodeID
            
            float v = Float.parseFloat(indicesAndValue[1]);
            int Ncop = Integer.MIN_VALUE;
            
            outputKey.set(indicesAndValue[0]);
            outputValue.set(Integer.toString(Ncop)+","+Float.toString(v));
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce_Input_BS extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String []value;
            float samples = Float.MIN_VALUE;
            int Ncopies = Integer.MIN_VALUE;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	samples = Math.max(Float.parseFloat(value[1]),samples);
            	Ncopies = Math.max(Integer.parseInt(value[0]),Ncopies);
            }
            
            context.write(null, new Text(key +","+ Integer.toString(Ncopies) +","+ Float.toString(samples)));
        }
    }
    
    
	//Job 1 - Phase 1 : Generate the bitonic sequence
    public static class Map1_BS extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            String iters = context.getConfiguration().get("iter");
            String steps = context.getConfiguration().get("step");
            String length = context.getConfiguration().get("length");
            String valConf = context.getConfiguration().get("val");
            String chunkConf = context.getConfiguration().get("chunk");
            
            //current step
            int MRstep = Integer.parseInt(iters); // map reduce step
            int step = Integer.parseInt(steps);   // main steps
            
            //max step = log2(total_input_size)
            int maxIter = (int) (Math.log(Integer.parseInt(length))/Math.log(2));
            
            int val = Integer.parseInt(valConf);
            int chunk = Integer.parseInt(chunkConf);
            
            int k = 0;
            int akey = Integer.parseInt(indicesAndValue[0]);
            String sep = null;
            int add = 0;
            int addmval = (int) (Math.pow(2,maxIter)-2);
            
            int side = 0;
            int rside = 0;
            
            if (MRstep == 1){
            	val = Integer.parseInt(length)/2;
            	
            	//generate new key
            	k = (int) Math.ceil(akey/2.0);
            	
            	//give a label "l" or "r" to each number.
            	int c = (int) Math.floor(akey/2) % 2;
            	
            	if (c == 1){
            		sep = "r";
            	}else{
            		sep = "l";
            	}
            	
            	if (step > 1){
            		side = (int) Math.ceil(akey/Math.pow(2, step));
            		
            		if (side % 2 == 1){
            			if (akey % 2 == 1){
            				sep = "l";
            			}else{
            				sep = "r";
            			}
            		}else{
            			if (akey % 2 == 1){
            				sep = "r";
            			}else{
            				sep = "l";
            			}
            		}
            	}
            }else if (MRstep == 2){
            	chunk = 4;
            	val = 4/2;
            	
            	if (akey <= chunk){
            		k = akey % val;
            		
            		if(akey > chunk/2){
            			sep = "r";
            		}else{
            			sep = "l";
            		}
            	}else{
            		add = (int) ((Math.ceil(akey/ (double)chunk) - 1) * addmval);
            		
            		k = akey % val + add;
            		
            		side = (int) Math.ceil(akey/Math.pow(2, step));
            		rside = (int) Math.ceil(akey/2.0);
            		
            		if (side % 2 == 1){
            			if (rside % 2 == 1){
            				sep = "l";
            			}else{
            				sep = "r";
            			}
            		}else{
            			if (rside % 2 == 1){
            				sep = "r";
            			}else{
            				sep = "l";
            			}
            		}
            	}
            }else{
            	val = (int) Math.pow(2,MRstep);
            	chunk = val;
            	val = val/2;
            	
            	if (akey <= chunk){
            		//generate new key
            		k = akey % val;
            		
            		//give a label "l" or "r" to each number.
            		if(akey > chunk/2){
            			sep = "r";
            		}else{
            			sep = "l";
            		}
            	}else{
            		
            		add = (int) ((Math.ceil(akey/ (double)chunk) - 1) * addmval);
            		
            		k = akey % val + add;
            		
            		//give a label "l" or "r" to each number.
            		side = (int) Math.ceil(akey/Math.pow(2, step));
            		rside = (int) Math.ceil(akey/(double)val);
            		
            		if (side % 2 == 1){
            			if (rside % 2 == 1){
            				sep = "l";
            			}else{
            				sep = "r";
            			}
            		}else{
            			if (rside % 2 == 1){
            				sep = "r";
            			}else{
            				sep = "l";
            			}
            		}
            	}
            }
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1] + "," + sep + "," + akey + ","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1_BS extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            
            float element1 = 0.0f;
            float element2 = 0.0f;
            float old_samples1 = 0.0f;
            float old_samples2 = 0.0f;
            int key1 = 0;
            int key2 = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	//check the label of the given number and store it.
            	if (value[1].equals("l")){
            		element1 = Float.parseFloat(value[0]);
            		key1 = Integer.parseInt(value[2]);
            		old_samples1 = Float.parseFloat(value[3]);
            	}else{
            		element2 = Float.parseFloat(value[0]);
            		key2 = Integer.parseInt(value[2]);
            		old_samples2 = Float.parseFloat(value[3]);
            	}
            }
            
            //number comparison
            if (element1 <= element2){
            	context.write(null, new Text(key1+ ","+Float.toString(element1)+ ","+Float.toString(old_samples1)));
            	context.write(null, new Text(key2+ ","+Float.toString(element2)+ ","+Float.toString(old_samples2)));
            }else{
            	context.write(null, new Text(key1+ ","+Float.toString(element2)+ ","+Float.toString(old_samples2)));
            	context.write(null, new Text(key2+ ","+Float.toString(element1)+ ","+Float.toString(old_samples1)));
            }
        }
    }
    
	//Job 2 - Phase 2 : Sort the bitonic sequence
    public static class Map2_BS extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            String iter = context.getConfiguration().get("iter");
            String length = context.getConfiguration().get("length");
            String valConf = context.getConfiguration().get("val");
            String chunkConf = context.getConfiguration().get("chunk");
            String offsetConf = context.getConfiguration().get("offset");
            
            //current step
            int MRstep = Integer.parseInt(iter);
            
            //max step = log2(total_input_size)
            int maxIter = (int) (Math.log(Integer.parseInt(length))/Math.log(2));
            int addmval = (int) (Math.pow(2,maxIter)-2);
            
            int val = Integer.parseInt(valConf);
            int chunk = Integer.parseInt(chunkConf);
            int offset = Integer.parseInt(offsetConf);
            
            int k = 0;
            int akey = Integer.parseInt(indicesAndValue[0]);
            String sep = null;
            int add = 0;
            
            if (MRstep == 1){
            	val = Integer.parseInt(length)/2;
            	
            	//generate new key
            	k = akey % val;
            	
        		//give a label "l" or "r" to each number.
            	if(akey > Float.parseFloat(length)/2.0){
            		sep = "r";
            	}else{
            		sep = "l";
            	}
            }else if (MRstep == maxIter){
            	k = (int) Math.ceil(akey/2.0);
            	
            	if (akey % 2 == 0){
            		sep = "r";
            	}else{
            		sep = "l";
            	}
            }else if (MRstep == 2){
            	if (akey <= chunk){
            		k = akey % val;
            		
            		if(akey > chunk/2){
            			sep = "r";
            		}else{
            			sep = "l";
            		}
            	}else{
            		k = akey % val + offset/2;
            		
            		if(akey > chunk/2 + offset){
            			sep = "r";
            		}else{
            			sep = "l";
            		}
            	}
            }else{
            	if (akey <= chunk){
            		k = akey % val;
            		
            		if(akey > chunk/2){
            			sep = "r";
            		}else{
            			sep = "l";
            		}
            	}else{
            		
            		add = (int) ((Math.ceil(akey/ (double)chunk) - 1) * addmval);
            		k = akey % val + add;
            		
            		if(akey > chunk/2 + offset*Math.ceil(akey/chunk) || akey % chunk == 0){
            			sep = "r";
            		}else{
            			sep = "l";
            		}
            	}
            }
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1] + "," + sep + "," + akey + ","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce2_BS extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            
            float element1 = 0.0f;
            float element2 = 0.0f;
            float old_samples1 = 0.0f;
            float old_samples2 = 0.0f;
            int key1 = 0;
            int key2 = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	if (value[1].equals("l")){
            		element1 = Float.parseFloat(value[0]);
            		key1 = Integer.parseInt(value[2]);
            		old_samples1 = Float.parseFloat(value[3]);
            	}else{
            		element2 = Float.parseFloat(value[0]);
            		key2 = Integer.parseInt(value[2]);
            		old_samples2 = Float.parseFloat(value[3]);
            	}
            }
            
            if (element1 >= element2){
            	context.write(null, new Text(key1+ ","+Integer.toString((int)element1)+ ","+Float.toString(old_samples1)));
            	context.write(null, new Text(key2+ ","+Integer.toString((int)element2)+ ","+Float.toString(old_samples2)));
            }else{
            	context.write(null, new Text(key1+ ","+Integer.toString((int)element2)+ ","+Float.toString(old_samples2)));
            	context.write(null, new Text(key2+ ","+Integer.toString((int)element1)+ ","+Float.toString(old_samples1)));
            }
        }
    }
    
    
    /***************************************
     * Redistribution map reduce functions *
     ***************************************/
    
	//Job 0 - generate nodeID
    public static class Map0_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String Nlength = context.getConfiguration().get("length");
            String iter = context.getConfiguration().get("iter");
            
            int nodeID = 0;
            int kin = Integer.parseInt(indicesAndValue[0]);
            double adj = ( Integer.parseInt(Nlength)/Math.pow(2, Integer.parseInt(iter)) );
            
            nodeID = (int) Math.ceil(kin/adj);
            
            outputValue.set(indicesAndValue[0]+","+indicesAndValue[1]+","+indicesAndValue[2]+","+nodeID);
            
            context.write(null, outputValue);
        }
    }
    
    /***************************
     * Compute cumsum(Ncopies) * 
     * *************************/
	
	//Job 1
    public static class Map1_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String iter = context.getConfiguration().get("iter");
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            
            if (Integer.parseInt(iter) == 0){
            	outputValue.set(indicesAndValue[1]+","+indicesAndValue[3]);
            }else{
            	outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            }
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1_Red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            int summation = 0;
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	summation += Integer.parseInt(value[0]);
            	nodeID = Integer.parseInt(value[1]);
            }
            
            context.write(null, new Text(key+","+Integer.toString(summation)+","+Integer.toString(nodeID)));
        }
    }
    
	//Job 2
    public static class Map2_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            outputKey.set(Integer.toString(kin));
            if (Float.parseFloat(iter) == Float.parseFloat(Nlength)-1.0f){
            	outputValue.set(indicesAndValue[1]+","+indicesAndValue[3]);
            }else{
            	outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            }
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Map22_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int kin = 2*Integer.parseInt(indicesAndValue[0]);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce2_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            //String adj = context.getConfiguration().get("adj");
            
            List<Integer> frvalues = new ArrayList<Integer>();
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	frvalues.add(Integer.parseInt(value[0]));
            	nodeID = Integer.parseInt(value[1]);
            }
            
            int rchild = Integer.parseInt(key.toString());
            int lchild = Integer.parseInt(key.toString())-1;
            
            if (Integer.parseInt(key.toString()) % 2 == 0){
            	if (Float.parseFloat(Nlength)-1 == Float.parseFloat(iter)){
            		context.write(null, new Text((lchild+nodeID)+ "," + Integer.toString(Collections.max(frvalues)-Collections.min(frvalues))+ "," + Integer.toString(nodeID)));
            		context.write(null, new Text((rchild+nodeID)+ "," + Integer.toString(Collections.max(frvalues))+ "," + Integer.toString(nodeID)));
            	}else{
            		context.write(null, new Text(lchild+ "," + Integer.toString(Collections.max(frvalues)-Collections.min(frvalues))+ "," + Integer.toString(nodeID)));
            		context.write(null, new Text(rchild+ "," + Integer.toString(Collections.max(frvalues))+ "," + Integer.toString(nodeID)));
            	}
            }
        }
    }
    
    /*************************************************************
     *   Add zeros so that we have the  cN = [0 cumsum(Ncopies)] *
     * ***********************************************************/
    
	//Job 3
    public static class Map24_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            outputKey.set(indicesAndValue[2]); 	 //nodeID
            outputValue.set(indicesAndValue[0]); //key
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce24_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            int max = Integer.MIN_VALUE;
            String adj = context.getConfiguration().get("adj");
            
            for (Text val : values) {
            	value = val.toString();
            	max = Math.max(Integer.parseInt(value),max);
            }
            
            int zvkey = max-Integer.parseInt(adj);
            
            //key = nodeID
            context.write(null, new Text(zvkey+","+Integer.toString(0)+","+key));
        }
    }
    
    /******************
     * Compute mindif * 
     * ****************/
	
	//Job 4
    public static class Map3_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String adjp1 = context.getConfiguration().get("adjp1");
            String hcNend = context.getConfiguration().get("hcNend");
            
            int ap1 = Integer.parseInt(adjp1);
            int haldadj = Integer.parseInt(hcNend)-1;
            int zeroval = Integer.parseInt(indicesAndValue[0]) % ap1;
            
            if ( zeroval <=  Integer.parseInt(hcNend) && zeroval != 0 ){
            	int nkey = Integer.parseInt(indicesAndValue[0])-(Integer.parseInt(indicesAndValue[2])-1);
            	if (Integer.parseInt(indicesAndValue[1]) <= haldadj){
                    outputValue.set(Integer.toString(nkey)+","+indicesAndValue[1]+","+indicesAndValue[2]);
                    context.write(null, outputValue);
            	}else{
                    outputValue.set(Integer.toString(nkey)+","+Integer.toString(haldadj)+","+indicesAndValue[2]);
                    context.write(null, outputValue);
            	}
            }
        }
    }
    
    /*******************************************
     *   Compute the Nfirst and Nraw via diff  *
     * *****************************************/
	
	//Job 5 - Generate the odd values of the diff function
    public static class Map4_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce4_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            List<Float> tvalues = new ArrayList<Float>();
            
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	tvalues.add(Float.parseFloat(value[0]));
            	nodeID = Integer.parseInt(value[1]);
            }
            
            //write the odd values of the final output
            //if (tvalues.size() > 1){
            if (tvalues.size() == 2){
            	int nkey = 2*Integer.parseInt(key.toString());
            	context.write(null, new Text(nkey-1 + ","+ Integer.toString((int)(Math.abs(tvalues.get(0)- tvalues.get(1))))+ ","+Integer.toString(nodeID)));
            }
        }
    }
    
	//Job 6 - Generate the even values of the diff function
    public static class Map5_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.floor(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce5_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            List<Float> tvalues = new ArrayList<Float>();
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	tvalues.add(Float.parseFloat(value[0]));
            	nodeID = Integer.parseInt(value[1]);
            }
            
            //write the even values of the final output
            if (tvalues.size() == 2){
                int nkey = 2*Integer.parseInt(key.toString())+1;
                context.write(null, new Text(nkey-1 + ","+Integer.toString((int)(Math.abs(tvalues.get(0)- tvalues.get(1))))+","+Integer.toString(nodeID)));
            }
        }
    }
    
    /*********************************
     *   compute oldsamples_first    *
     * *******************************/
	
	//Job 7 - Generate the oldsamples vector
    public static class Map6_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            //Text outputKey = new Text();
            Text outputValue = new Text();
            
            String adjp1 = context.getConfiguration().get("adjp1");
            String hcNend = context.getConfiguration().get("hcNend");
            
            int ap1 = Integer.parseInt(adjp1)-1; //adj
            int haldadj = Integer.parseInt(hcNend)-1; //adjhalf-1
            int zeroval = Integer.parseInt(indicesAndValue[0]) % ap1;
            
            if ( zeroval <=  haldadj && zeroval != 0){
            	outputValue.set(indicesAndValue[0]+","+indicesAndValue[2]);
            	context.write(null, outputValue);
            }
        }
    }
    
	
    /************************
     *   Compute the maxdif *
     * **********************/
    
    //Job 8
    public static class Map7_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            String adjp1 = context.getConfiguration().get("adjp1");
            String hcNend = context.getConfiguration().get("hcNend");
            
            int ap1 = Integer.parseInt(adjp1);
            int haldadj = Integer.parseInt(hcNend)-1;
            int zeroval = Integer.parseInt(indicesAndValue[0]) % ap1;
            
            if ( zeroval <=  Integer.parseInt(hcNend) && zeroval != 0 ){
            	if (Integer.parseInt(indicesAndValue[1]) >= haldadj){
                    outputValue.set(indicesAndValue[0]+","+indicesAndValue[1]+","+indicesAndValue[2]);
                    context.write(null, outputValue);
            	}else{
                    outputValue.set(indicesAndValue[0]+","+Integer.toString(haldadj)+","+indicesAndValue[2]);
                    context.write(null, outputValue);
            	}
            }
        }
    }
    
    /*******************
     *   Compute slef  *
     * *****************/
    
	//Job 9 - compute slef = cN(2:(N/2+1))<=N/2
    public static class Map8_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            
            String adjp1 = context.getConfiguration().get("adjp1");
            String hcNend = context.getConfiguration().get("hcNend");
            
            int ap1 = Integer.parseInt(adjp1);
            int haldadj = Integer.parseInt(hcNend)-1;
            int zeroval = Integer.parseInt(indicesAndValue[0]) % ap1;
            
            if ( zeroval >=2 && zeroval <=  Integer.parseInt(hcNend)){
            	if (Integer.parseInt(indicesAndValue[1]) <= haldadj){
                    outputValue.set(indicesAndValue[0]+","+Integer.toString(1)+","+indicesAndValue[2]);
                    context.write(null, outputValue);
            	}else{
                    outputValue.set(indicesAndValue[0]+","+Integer.toString(0)+","+indicesAndValue[2]);
                    context.write(null, outputValue);
            	}
            }
        }
    }
    
    /*******************
     *   Compute nrig  *
     *******************/
    
	//Job 11 - compute nrig = cN(2:(N/2+1))>0
    public static class Map10_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            
            String adjp1 = context.getConfiguration().get("adjp1");
            String hcNend = context.getConfiguration().get("hcNend");
            
            int ap1 = Integer.parseInt(adjp1)-1;
            int zeroval = Integer.parseInt(indicesAndValue[0]) % ap1;
            
            if ( zeroval >= Integer.parseInt(hcNend) || zeroval == 0){
            	if (Integer.parseInt(indicesAndValue[1]) > 0){
                    outputValue.set(indicesAndValue[0]+","+Integer.toString(1)+","+indicesAndValue[3]);
                    context.write(null, outputValue);
            	}else{
                    outputValue.set(indicesAndValue[0]+","+Integer.toString(0)+","+indicesAndValue[3]);
                    context.write(null, outputValue);
            	}
            }
        }
    }
    
    /*******************************
     *   Compute nleft and nright  *
     * *****************************/
    
	//Job 10 - summation
    
    //nrig map
    public static class Map9_1_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    //slef map
    public static class Map9_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            if (Integer.parseInt(indicesAndValue[2]) % 2 == 0){
            	k = (int) Math.ceil(kin/2.0);
            }else{
            	k = (int) Math.floor(kin/2.0);
            }
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce9_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            float summation = 0.0f;
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	summation += Float.parseFloat(value[0]);
            	nodeID = Integer.parseInt(value[1]);
            }
            
            context.write(null, new Text(key+ ","+Float.toString(summation)+","+Integer.toString(nodeID)));
        }
    }
    
    /************************************************************
     *   give the correct keys to the nleft and nright vectors  *
     *   based on the nodeID.                                   *
     * **********************************************************/
	
    public static class Map9_nright_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - value
            //indicesAndValue[2] - nodeID
            
            float v = Float.parseFloat(indicesAndValue[1]);
            
            outputKey.set(indicesAndValue[2]);
            outputValue.set(Float.toString(v));
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Map9_nleft_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - value
            //indicesAndValue[2] - nodeID
            
            float v = Float.parseFloat(indicesAndValue[1]);
            
            outputKey.set(indicesAndValue[2]);
            outputValue.set(Float.toString(v));
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce_nlr_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float mval = 0.0f;
            
            for (Text val : values) {
            	value = val.toString();
            	mval = Float.parseFloat(value) - mval;
            }
            
            mval = Math.abs(mval);
            context.write(null, new Text(key+ ","+Integer.toString((int)mval)));
        }
    }
    
    
    /****************************
     *   combine nlr and in0 *
     * **************************/
	
    public static class Map9_in0_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //nodeID
            outputKey.set(indicesAndValue[3]);
            
            //nlr+ncopies+oldsamples+key
            outputValue.set("0"+","+indicesAndValue[1]+","+indicesAndValue[2]+","+indicesAndValue[0]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Map9_nlr_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //nodeID
            outputKey.set(indicesAndValue[0]);
            
            //nlr+ncopies+oldsamples+nodeID
            outputValue.set(indicesAndValue[1]+","+"-1"+","+"-1"+","+"-1");
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce_input00_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            
            List<Integer> newkey1 = new ArrayList<Integer>();
            List<Integer> ncopies1 = new ArrayList<Integer>();
            List<Float> oldsamples1 = new ArrayList<Float>();
            
            //nodeID = key
            int nlr = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	ncopies1.add(Integer.parseInt(value[1]));
            	oldsamples1.add(Float.parseFloat(value[2]));
            	newkey1.add(Integer.parseInt(value[3]));
            	
            	nlr += Integer.parseInt(value[0]);
            }
            
            for(int i=0;i < oldsamples1.size(); i++){
            	if(oldsamples1.get(i)!=-1){
            		context.write(null, new Text(Integer.toString(newkey1.get(i))+","+Integer.toString(ncopies1.get(i))+","+Float.toString(oldsamples1.get(i))+","+key+","+Integer.toString(nlr)));
            	}
            }
            
        }
    }
    
    /***********************************************************
     *   compute index = mod((nleft-nright)+(1:N/2)-1,N/2)+1;  *
     *   and	 olpsamples2 = oldsamples(index);              *
     * *********************************************************/
	
    //Job 9
    //compute index = mod((nleft-nright)+(1:N/2)-1,N/2)+1;
    //compute olpdsamples2 = oldsamples(index);       
    public static class Map11_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            
          	//indicesAndValue[0] --  key    	 -- (integer)
        	//indicesAndValue[1] --  ncopied 	 -- (integer)
            //indicesAndValue[2] --  oldsamples  -- (float)
            //indicesAndValue[3] --  nodeID 	 -- (integer)
        	//indicesAndValue[4] --  nlr         -- (integer)
            
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String halfNtemp = context.getConfiguration().get("halfN");
            String adjtemp = context.getConfiguration().get("adj");
            
            int halfN = Integer.parseInt(halfNtemp); 		//N/2
            int adj = Integer.parseInt(adjtemp); 			//size of vector
            int kin = Integer.parseInt(indicesAndValue[0]); //key
            int valid = kin % adj; //is used to check if the key is in the interval of [1,N/2]
            
            if (valid >= 1 && valid <= halfN){
            	//kin = ((nrl + kin -1) % halfN + 1) + (nodeID-1) * adj
            	int oldkey = kin;
            	kin = ((Integer.parseInt(indicesAndValue[4]) + kin -1) % halfN + 1)+(Integer.parseInt(indicesAndValue[3])-1)*adj; 
            	
                outputKey.set(Integer.toString(kin));
                outputValue.set(Integer.toString(oldkey)+","+indicesAndValue[2]+","+Integer.toString(kin)+","+indicesAndValue[3]+","+indicesAndValue[4]);
                context.write(null, outputValue);
            }
        }
    }
    
    //input00 is the input
    public static class Map11_1_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            
          	//indicesAndValue[0] --  key    	    -- (integer)
            //indicesAndValue[1] --  ncopies        -- (integer)
            //indicesAndValue[2] --  oldsamples     -- (integer)
        	//indicesAndValue[3] --  nodeID         -- (integer)
            //indicesAndValue[4] --  nlr            -- (integer)
            
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            outputKey.set(indicesAndValue[0]);
            outputValue.set("-1"+","+indicesAndValue[2]+","+indicesAndValue[3]);
            context.write(outputKey, outputValue);
        }
    }
    
    //oldsamples2 as input
    public static class Map11_2_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            
          	//indicesAndValue[0] --  oldkey    	    -- (integer)
            //indicesAndValue[1] --  oldsamples     -- (float)
            //indicesAndValue[2] --  newkey 	    -- (integer)
        	//indicesAndValue[3] --  nodeID         -- (integer)
            //indicesAndValue[4] --  nlr            -- (integer)
            
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            outputKey.set(indicesAndValue[2]);
            outputValue.set(indicesAndValue[0]+","+"-1.0"+","+indicesAndValue[3]);
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce_oldsamples2_old2_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            
            //List<Integer> k = new ArrayList<Integer>();
            float v = Float.MIN_VALUE;
            int nodeID = 0;
            int k = Integer.MIN_VALUE;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	k = Math.max(Integer.parseInt(value[0]),k); //key
            	v = Math.max(Float.parseFloat(value[1]),v); //value
            	nodeID = Integer.parseInt(value[2]); //nodeID
            }

            if (k >= 0){
            	 context.write(null, new Text(Integer.toString(k)+","+Float.toString(v)+","+Integer.toString(nodeID)));
            }
         }
    }
    
    /*****************************************
     *   compute fin = cN(2:(N/2+1))>N/2     *
     * ***************************************/
	
	//Job 10 - compute f  = cN(2:(N/2+1))>N/2
    public static class Map12_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            
            String adjp1 = context.getConfiguration().get("adjp1");
            String hcNend = context.getConfiguration().get("hcNend");
            
            int ap1 = Integer.parseInt(adjp1);
            int haldadj = Integer.parseInt(hcNend)-1;
            int zeroval = Integer.parseInt(indicesAndValue[0]) % ap1;
            
            if ( zeroval >=2 && zeroval <=  Integer.parseInt(hcNend)){
            	if (Integer.parseInt(indicesAndValue[1]) > haldadj){
                    outputValue.set(indicesAndValue[0]+","+Integer.toString(1)+","+indicesAndValue[2]);
                    context.write(null, outputValue); // key | value | nodeID
            	}else{
                    outputValue.set(indicesAndValue[0]+","+Integer.toString(0)+","+indicesAndValue[2]);
                    context.write(null, outputValue); // key | value | nodeID
            	}
            }
        }
    }
    
    /**************************************
     *   compute f = fin(index)           *
     *   compute NrawIndexx = Nraw(index) *
     * ************************************/
	
    public static class Map19_fin_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //fin
            //indicesAndValue[0] - newkey
            //indicesAndValue[1] - value
            //indicesAndValue[2] - nodeID
            
            //nodeID
            outputKey.set(indicesAndValue[0]);
            
            //newkey+val
            outputValue.set("-1"+","+indicesAndValue[1]+","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Map19_nraw_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //fin
            //indicesAndValue[0] - newkey
            //indicesAndValue[1] - value
            //indicesAndValue[2] - nodeID
            int nkey = Integer.parseInt(indicesAndValue[0])+1;
            
            //nodeID
            outputKey.set(Integer.toString(nkey));
            
            //newkey+val
            outputValue.set("-1"+","+indicesAndValue[1]+","+indicesAndValue[2]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Map19_oldsamples2_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputValue = new Text();
            Text outputKey = new Text();
            
            //oldsamples2
            //indicesAndValue[0] - newkey
            //indicesAndValue[1] - value
            //indicesAndValue[2] - oldkey
            //indicesAndValue[3] - nodeID
            
            //the key_fin = key_oldsamples2 plus nodeID
            int keyfin = Integer.parseInt(indicesAndValue[2])+Integer.parseInt(indicesAndValue[3]);
            outputKey.set(Integer.toString(keyfin));
            
            //newkey plus nodeID
            int newkeyf = Integer.parseInt(indicesAndValue[0])+Integer.parseInt(indicesAndValue[3]);
            
            //newkey+val
            outputValue.set(Integer.toString(newkeyf)+","+"-1"+","+indicesAndValue[3]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce_f_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            
            //newkey
            int v = Integer.MIN_VALUE;
            int newkey = Integer.MIN_VALUE;
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	v = Math.max(Integer.parseInt(value[1]),v); //value
            	newkey = Math.max(Integer.parseInt(value[0]),newkey); //newkey
            	nodeID = Integer.parseInt(value[2]); //nodeID
            }
            
            //key+val
            context.write(null, new Text(Integer.toString(newkey)+","+Integer.toString(v)+","+Integer.toString(nodeID)));
        }
    }
    
    /**************************************************
     * ex: compute old1 = ~f.*oldsamples(:,(N/2+1):N) * 
     * ************************************************/
	
    //f
    public static class Map13_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - value
            //indicesAndValue[2] - nodeID
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            
            outputKey.set(Integer.toString(kin));
            
            //create the ~f
            if (Float.parseFloat(indicesAndValue[1]) == 1.0f){
            	outputValue.set(Integer.toString(0)+","+indicesAndValue[2]);
            }else{
            	outputValue.set(Integer.toString(1)+","+indicesAndValue[2]);
            }
            
          	context.write(outputKey, outputValue);
        }
    }
    
	//oldsamples
    public static class Map14_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String hcNend = context.getConfiguration().get("hcNend"); // N/2+1
            String N = context.getConfiguration().get("adj");
            
          	//indicesAndValue[0] --  key    	 -- (integer)
        	//indicesAndValue[1] --  ncopied 	 -- (integer)
            //indicesAndValue[2] --  oldsamples  -- (float)
            //indicesAndValue[3] --  nodeID 	 -- (integer)
        	//indicesAndValue[4] --  nlr         -- (integer)
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            //N/2
            int nk = Integer.parseInt(hcNend) - 1; // N/2
            int nodeID = Integer.parseInt(indicesAndValue[3]);
            
            int zeroval = Integer.parseInt(indicesAndValue[0]) % Integer.parseInt(N);
            
            //hcNend = N/2 + 1
            //key = 2:hcNend
            if ( zeroval >= Integer.parseInt(hcNend) || zeroval == 0){
            	
            	int nkey = kin+nodeID-nk;
            	
                outputKey.set(Integer.toString(nkey));
                outputValue.set(indicesAndValue[2]+","+indicesAndValue[3]);
                context.write(outputKey, outputValue);
            }
            
        }
    }
    
	//ncopies
    public static class Map14_ncopies_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            String hcNend = context.getConfiguration().get("hcNend"); // N/2+1
            String N = context.getConfiguration().get("adj");
            
          	//indicesAndValue[0] --  key    	 -- (integer)
        	//indicesAndValue[1] --  ncopied 	 -- (integer)
            //indicesAndValue[2] --  oldsamples  -- (float)
            //indicesAndValue[3] --  nodeID 	 -- (integer)
        	//indicesAndValue[4] --  nlr         -- (integer)
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            //N/2
            int nk = Integer.parseInt(hcNend) - 1; // N/2
            int nodeID = Integer.parseInt(indicesAndValue[3]);
            
            int zeroval = Integer.parseInt(indicesAndValue[0]) % Integer.parseInt(N);
            
            if ( zeroval >= Integer.parseInt(hcNend) || zeroval == 0){
            	
            	int nkey = kin+nodeID-nk;
            	
                outputKey.set(Integer.toString(nkey));
                outputValue.set(indicesAndValue[1]+","+indicesAndValue[3]);
                context.write(outputKey, outputValue);
            }
            
        }
    }
    
    public static class Reduce13_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            float prd = 1.0f;
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	prd *= Float.parseFloat(value[0]);
            	nodeID = Integer.parseInt(value[1]);
            }
            
            context.write(null, new Text(key + "," + Float.toString(prd)+","+Integer.toString(nodeID)));
        }
    }
    
    /***************************************
     *  ex: compute old2 = f.*oldsamples2  * 
     * *************************************/
	
    //f
    public static class Map15_red extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
          	context.write(outputKey, outputValue);
        }
    }
    
    public static class Map16_oldsamples2_old2_red extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
          	//indicesAndValue[0] --  oldkey    	   -- (integer) -- this is not used and not needed
        	//indicesAndValue[1] --  oldsamples    -- (float)
            //indicesAndValue[2] --  newkey  	   -- (integer)
            //indicesAndValue[3] --  nodeID        -- (integer)
            
            int kin = Integer.parseInt(indicesAndValue[0])+Integer.parseInt(indicesAndValue[2]);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Map16_nrawindex_red extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce14_red extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            float prd = 1.0f;
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	prd *= Float.parseFloat(value[0]);
            	nodeID = Integer.parseInt(value[1]);
            }
            
            context.write(null, new Text(key + "," + Float.toString(prd)+","+Integer.toString(nodeID)));
        }
    }
    
    /***********************************************
     *  ex: compute oldsamples_second = old1+old2  *
     * *********************************************/
	
    public static class Map17_red extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
          	context.write(outputKey, outputValue);
        }
    }
    
    public static class Map18_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            outputKey.set(Integer.toString(kin));
            outputValue.set(indicesAndValue[1]+","+indicesAndValue[2]);
            context.write(outputKey, outputValue);
        }
    }
    
    //oldsamples_second
    public static class Reduce15_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        	String N = context.getConfiguration().get("adj");
        	String[] value;
            float sum = 0.0f;
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	sum += Float.parseFloat(value[0]);
            	nodeID = Integer.parseInt(value[1]);
            }
            
            int nkey = Integer.parseInt(key.toString())+Integer.parseInt(N)/2-1-(nodeID-1);
            
            context.write(null, new Text(Integer.toString(nkey) + "," + Float.toString(sum)));
        }
    }
    
    //Nsecond
    public static class Reduce15_Nsecond_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        	String N = context.getConfiguration().get("adj");
        	String[] value;
            float sum = 0.0f;
            int nodeID = 0;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	sum += Float.parseFloat(value[0]);
            	nodeID = Integer.parseInt(value[1]);
            }
            
            int nkey = Integer.parseInt(key.toString())+Integer.parseInt(N)/2-1-(nodeID-1);
            
            context.write(null, new Text(Integer.toString(nkey) + "," + Integer.toString((int)sum)));
        }
    }
    

    /***************************************************************************************
     *  Merge the Nfirst, Nsecond, oldsamples_first, oldsamples_second for the next level  *
     *  This is used as an input for the next level of the tree.                           * 
     * *************************************************************************************/
	
    public static class Map25_NewNcopies_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - ncopies
            
            float newsamplesVal = -1.0f;
            
            outputKey.set(indicesAndValue[0]);
            outputValue.set(Float.toString(newsamplesVal)+","+indicesAndValue[1]); //newsamples(not real values) + newncopies
          	context.write(outputKey, outputValue);
        }
    }
    
    public static class Map25_NewOldsamples_red extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int newncopies = 0;
            
            outputKey.set(indicesAndValue[0]);
            outputValue.set(indicesAndValue[1]+","+Integer.toString(newncopies)); //newsamples + newncopies(not real values)
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce25_NewInput_red extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            int newncopies = Integer.MIN_VALUE;;
            float newsamples = Float.MIN_VALUE;
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	newncopies = Math.max(Integer.parseInt(value[1]),newncopies);
            	newsamples = Math.max(Float.parseFloat(value[0]),newsamples);
            }
            
            context.write(null, new Text(key + "," + Integer.toString(newncopies)+","+Float.toString(newsamples)));
        }
    }
    
    
    
    
    public static class Map_vec_particles extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - value
            
            outputKey.set(indicesAndValue[0]);
            outputValue.set(indicesAndValue[1]);
          	context.write(outputKey, outputValue);
        }
    }
    
    /********
     * xest *
     ********/
    
    // vector vector multiplication
    public static class Map_vec_weights extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            //indicesAndValue[0] - key
            //indicesAndValue[1] - value
            
            outputKey.set(indicesAndValue[0]);
            outputValue.set(indicesAndValue[1]);
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce_multiplication extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float mul = 1.0f;
            
            for (Text val : values) {
            	value = val.toString();
            	
            	mul *= Float.parseFloat(value);
            }
            
            context.write(null, new Text(key + "," + mul));
        }
    }
    
	//Job 1 - parallel summation
    public static class Map1xest extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1xest extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float summation = 0.0f;
            
            for (Text val : values) {
            	value = val.toString();
            	summation += Float.parseFloat(value);
            }
            
            context.write(null, new Text(key+","+Float.toString(summation)));
        }
    }
    
    
    /*********
     * xestm *
     *********/
    
	//Job 1 - parallel summation
    public static class Map1xestm extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1] + "," + indicesAndValue[2] );
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1xestm extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String[] value;
            float summation = 0.0f;
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            for (Text val : values) {
            	value = val.toString().split(",");
            	
            	summation += Float.parseFloat(value[1]);
            }
            
            if (Math.log(Float.parseFloat(Nlength))/Math.log(2) == Float.parseFloat(iter)+1){
            	float val = summation/Float.parseFloat(Nlength);
            	context.write(null, new Text(key+","+ Float.toString(1) +","+Float.toString(val)));
            }else{
            	context.write(null, new Text(key+","+ Float.toString(1) +","+Float.toString(summation)));
            }
        }
    }
    
	//Job 1 - parallel summation
    public static class Map1xestmF extends Mapper<LongWritable, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] indicesAndValue = line.split(",");
            Text outputKey = new Text();
            Text outputValue = new Text();
            
            int k;
            int kin = Integer.parseInt(indicesAndValue[0]);
            
            k = (int) Math.ceil(kin/2.0);
            
            outputKey.set(Integer.toString(k));
            outputValue.set(indicesAndValue[1]);
            
            context.write(outputKey, outputValue);
        }
    }
    
    public static class Reduce1xestmF extends Reducer<Text, Text, Text, Text> {
        
        private String foo;
        
        @Override
        public void setup(Context context) {
            foo = context.getConfiguration().get("foo");
            if (StringUtils.isEmpty(foo)) {
                foo = "DEFAULT";
            }
            System.out.println("foo equals " + foo);
        }
        
        @Override
    	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String value;
            float summation = 0.0f;
            String Nlength = context.getConfiguration().get("Size");
            String iter = context.getConfiguration().get("iter");
            
            for (Text val : values) {
            	value = val.toString();
            	
            	summation += Float.parseFloat(value);
            }
            
            if (Math.log(Float.parseFloat(Nlength))/Math.log(2) == Float.parseFloat(iter)+1){
            	float val = summation/Float.parseFloat(Nlength);
            	context.write(null, new Text(key +","+Float.toString(val)));
            }else{
            	context.write(null, new Text(key +","+Float.toString(summation)));
            }
        }
    }
    
    /***************************
     * Random number generator *
     ***************************/
    
    //random number between [0.001, 0.9999]
    public static float randFloat(float min, float max) {
    	
        // NOTE: Usually this should be a field rather than a method
        // variable so that it is not re-seeded every call.
        Random rand = new Random();
        
        // nextInt is normally exclusive of the top value,
        // so add 1 to make it inclusive
        float randomNum = (max - min)* rand.nextFloat() + min;
        
        return randomNum;
    }
    
    /***********************************
     * Delete Directories or/and Files *
     ***********************************/
    
    public static boolean deleteDirectory(File directory) {
        if(directory.exists()){
            File[] files = directory.listFiles();
            if(null!=files){
                for(int i=0; i<files.length; i++) {
                	String name = files[i].getName().toString();
                	
                	if (name != "2"){
                        if(files[i].isDirectory()) {
                            deleteDirectory(files[i]);
                        }
                        else {
                            files[i].delete();
                        }
                	}
                }
            }
        }
        
        return(directory.delete()); 
        
    }
    
    /*************************
     * Weights Normalization 
     * @param conf1List *
     *************************/
    
    public static void weightsNormalization(Configuration[] confList) throws IOException, ClassNotFoundException, InterruptedException{
        
    	String input = "";
    	String output = "";
    	
    	//Job - find max(w)
    	for (int i=0;i<confList.length;i++){
        	Job job1 = Job.getInstance(confList[i], "Job1");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map1max_weig.class);
        	job1.setReducerClass(Reduce1max_weig.class);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "vec_Weights";
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "vec_Weights" + i;
        	}
        	
        	// setting the output file
        	output = "vec_Weights" + (i + 1);
        	
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
    	}
    	
    	//retrieve max(x)
		float maxweights = 0.0f;
        try {
			BufferedReader br = new BufferedReader(new FileReader(output+"/part-r-00000"));
			
			String line = null;
    		while ((line = br.readLine()) != null) {
    			String[] val = line.split(",");
    			maxweights = Float.parseFloat(val[1]);
    		}
    		
    		br.close();
        } catch (IOException e) {
            System.out.println("File Read Error");
        }
        
    	//compute w = w - max(w)
		Configuration conf1red = new Configuration();
		conf1red.set("maxw", Float.toString(maxweights));
		
    	Job job2 = Job.getInstance(conf1red, "Job1");
    	job2.setJarByClass(PF_Hadoop.class);
    	job2.setOutputKeyClass(Text.class);
    	job2.setOutputValueClass(Text.class);
    	
    	job2.setMapperClass(Map2_weig.class);
    	job2.setNumReduceTasks(0);
    	
    	job2.setInputFormatClass(TextInputFormat.class);
    	job2.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "vec_Weights";
    	output = "vec_WeightsM";
    	
    	TextInputFormat.addInputPath(job2, new Path(input));
    	TextOutputFormat.setOutputPath(job2, new Path(output));
        
    	job2.waitForCompletion(true);
        
    	
    	//compute log(sum(exp(w)))
    	for (int i=0;i<confList.length;i++){
        	Job job1 = Job.getInstance(confList[i], "Job1");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map3_weig.class);
        	job1.setReducerClass(Reduce3_weig.class);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "vec_WeightsM";
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "vec_WeightsM" + i;
        	}
        	
        	// setting the output file
        	output = "vec_WeightsM" + (i + 1);
        	
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
    	}
    	
    	//retrieve  log(sum(exp(w)))
		float lgsmexpw = 0.0f;
        try {
			BufferedReader br = new BufferedReader(new FileReader(output+"/part-r-00000"));
			
			String line = null;
    		while ((line = br.readLine()) != null) {
    			String[] val = line.split(",");
    			lgsmexpw = Float.parseFloat(val[1]);
    		}
    		
    		br.close();
        } catch (IOException e) {
            System.out.println("File Read Error");
        }
        
        
    	//compute w = w -log(sum(exp(w)))
		Configuration conf3 = new Configuration();
		conf3.set("lgsmexpw", Float.toString(lgsmexpw));
		
    	Job job3 = Job.getInstance(conf3, "Job3");
    	job3.setJarByClass(PF_Hadoop.class);
    	job3.setOutputKeyClass(Text.class);
    	job3.setOutputValueClass(Text.class);
    	
    	job3.setMapperClass(Map4_weig.class);
    	job3.setNumReduceTasks(0);
    	
    	job3.setInputFormatClass(TextInputFormat.class);
    	job3.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "vec_WeightsM";
    	output = "weights_output";
    	
    	TextInputFormat.addInputPath(job3, new Path(input));
    	TextOutputFormat.setOutputPath(job3, new Path(output));
        
    	job3.waitForCompletion(true);
        
    	
        /********************
         *  File Management * 
         * ******************/
    	
    	FileSystem fs = FileSystem.get(conf3);
    	FileStatus[] fileStatus = fs.listStatus(new Path("."));
    	Path[] paths = FileUtil.stat2Paths(fileStatus);
    	for (Path path : paths){
    		if (!(path.getName().equals("src")) && !(path.getName().equals("bin")) && !(path.getName().equals(".settings")) && !(path.getName().equals("weights_output")) && !(path.getName().equals("new_state")) && !(path.getName().equals(".classpath")) && !(path.getName().equals(".project"))){
    			fs.delete(path, true);
    		}
    	}
    	
    	
		fs.rename(new Path("weights_output"), new Path("input0"));
    	
    }
    
    /*************************
     * Effective Sample Size 
     * @param confList *
     *************************/
    
    public static float effsamplesize(Configuration[] confList) throws IOException, ClassNotFoundException, InterruptedException{
    	
    	String input= "";
    	String output = "";
    	
    	//compute log(sum(exp(w)))
    	for (int i=0;i<confList.length;i++){
        	Job job1 = Job.getInstance(confList[i], "Job1N");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map1_Neff.class);
        	job1.setReducerClass(Reduce1_Neff.class);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "input0";
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "input0" + i;
        	}
        	
        	// setting the output file
        	output = "input0" + (i + 1);
        	
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
    	}
    	
    	//retrieve sum(exp(w.^2))
		float smexw2 = 0.0f;
        try {
			BufferedReader br = new BufferedReader(new FileReader(output+"/part-r-00000"));
			
			String line = null;
    		while ((line = br.readLine()) != null) {
    			String[] val = line.split(",");
    			smexw2 = Float.parseFloat(val[1]);
    		}
    		
    		br.close();
        } catch (IOException e) {
            System.out.println("File Read Error");
        }
        
        /********************
         *  File Management * 
         * ******************/
    	
    	FileSystem fs = FileSystem.get(confList[0]);
    	FileStatus[] fileStatus = fs.listStatus(new Path("."));
    	Path[] paths = FileUtil.stat2Paths(fileStatus);
    	for (Path path : paths){
    		if (!(path.getName().equals("src")) && !(path.getName().equals("bin")) && !(path.getName().equals(".settings")) && !(path.getName().equals("input0")) && !(path.getName().equals("new_state")) && !(path.getName().equals(".classpath")) && !(path.getName().equals(".project"))){
    			fs.delete(path, true);
    		}
    	}
		
		//return Neff
		float Neff = 1.0f/smexw2;
		
    	return Neff;
    }
    
    
    
    /*******************************
     * Minimum Variance Resampling 
     * @param confList 
     * @throws IOException 
     * @throws IllegalArgumentException 
     * @throws InterruptedException 
     * @throws ClassNotFoundException *
     *******************************/
    
    public static void mvr(int Nlength, Configuration[] confList) throws IllegalArgumentException, IOException, InterruptedException, ClassNotFoundException{
    	
        /*******************************************
         *      Compute the cumulative summation   *
         * *****************************************/
        
    	String input = "";
    	String output = "";
    	
    	
    	//Job 1 - Forward : Parallel Summation
    	for (int i=0;i<confList.length;i++){
        	Job job1 = Job.getInstance(confList[i], "Job1");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map1.class);
        	job1.setReducerClass(Reduce1.class);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "input0";
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "input0" + i;
        	}
        	
        	// setting the output file
        	output = "input0" + (i + 1);
        	
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
        	
    	}
    	
    	int maxN = (int) confList.length;
    	String iterOutput = "comb" + (maxN-1) + ".txt";
    	
    	//copy files : copy two files in one
    	ProcessBuilder pb = new ProcessBuilder("cat", "input0"+Integer.toString(maxN)+"/part-r-00000","input0"+Integer.toString(maxN-1)+"/part-r-00000");
    	File bias = new File(iterOutput);
    	pb.redirectOutput(Redirect.appendTo(bias));
    	pb.start();
    	
    	int j=maxN-2;
    	String comb = null; 
    	
    	//Job 2 - Backward : Parallel "Subtract"
    	for (int i=0;i<confList.length;i++){
        	Job job2 = Job.getInstance(confList[i], "Job2");
        	job2.setJarByClass(PF_Hadoop.class);
        	job2.setOutputKeyClass(Text.class);
        	job2.setOutputValueClass(Text.class);
        	
        	job2.setMapperClass(Map2.class);
        	job2.setReducerClass(Reduce2.class);
        	
        	job2.setInputFormatClass(TextInputFormat.class);
        	job2.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0){
        		input = iterOutput;
        	}else{
        		input = comb;
        	}
        	
        	output = "out"+ Integer.toString(j);
        	
        	TextInputFormat.addInputPath(job2, new Path(input));
        	TextOutputFormat.setOutputPath(job2, new Path(output));
            
        	job2.waitForCompletion(true);
        	
        	comb = "comb"+Integer.toString(j)+".txt";
        	
        	String sCopy = null;
        	if (j == 0){
        		sCopy = "input"+ Integer.toString(j)  +"/part-r-00000";
        	}else{
        		sCopy = "input0"+ Integer.toString(j)  +"/part-r-00000";
        	}
        	
        	//copy files : copy two files in one
        	ProcessBuilder pb1 = new ProcessBuilder("cat", output+"/part-r-00000", sCopy);
        	File bias1 = new File(comb);
        	pb1.redirectOutput(Redirect.appendTo(bias1));
        	pb1.start();
        	j--;
    	}
    	
        /**************************************************
         *   Compute the "floor(cumsum(w)*N/w(end)+rand)" *
         * ************************************************/
        
        float wend = 0.0f;
        
        //retrieve wend
        try{
        	BufferedReader read = new BufferedReader(new FileReader("input0" + Integer.toString((int)confList.length) + "/part-r-00000"));
        	String str = read.readLine();
        	String[] r = str.split(",");
        	wend = Float.parseFloat(r[1]);
        	read.close();
        }catch (IOException e) {
        	System.out.println("File Error");
        }
    	
        //compute Nwend = N/w(end)
        float Nwend = (float) (Nlength/wend);
        
        //random number between [min,max) = [0.001, 1)
        float randNum = randFloat(0.001f, 1.0f);
        
        //Job 3
		Configuration conf3 = new Configuration();
		conf3.set("randNum", Float.toString(randNum));
		conf3.set("Nwend", Float.toString(Nwend));
		
    	Job job3 = Job.getInstance(conf3, "Job3");
    	job3.setJarByClass(PF_Hadoop.class);
    	job3.setOutputKeyClass(Text.class);
    	job3.setOutputValueClass(Text.class);
    	
    	job3.setMapperClass(Map3.class);
    	//job3.setReducerClass(Reduce3.class);
    	job3.setNumReduceTasks(0);
    	
    	job3.setInputFormatClass(TextInputFormat.class);
    	job3.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "out-1/part-r-00000";
    	output = "output";
    	
    	TextInputFormat.addInputPath(job3, new Path(input));
    	TextOutputFormat.setOutputPath(job3, new Path(output));
        
    	job3.waitForCompletion(true);
    	
        /**********************
         *   Compute the diff *
         * ********************/
        
    	String inputDiff = output;
    	
        //Job 4
    	Job job4 = Job.getInstance(confList[0], "Job4");
    	job4.setJarByClass(PF_Hadoop.class);
    	job4.setOutputKeyClass(Text.class);
    	job4.setOutputValueClass(Text.class);
    	
    	job4.setMapperClass(Map4.class);
    	job4.setReducerClass(Reduce4.class);
    	
    	job4.setInputFormatClass(TextInputFormat.class);
    	job4.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = inputDiff;
    	output = "output1diff";
    	
    	TextInputFormat.addInputPath(job4, new Path(input));
    	TextOutputFormat.setOutputPath(job4, new Path(output));
        
    	job4.waitForCompletion(true);
        
        //Job 5
    	Job job5 = Job.getInstance(confList[0], "Job5");
    	job5.setJarByClass(PF_Hadoop.class);
    	job5.setOutputKeyClass(Text.class);
    	job5.setOutputValueClass(Text.class);
    	
    	job5.setMapperClass(Map5.class);
    	job5.setReducerClass(Reduce5.class);
    	
    	job5.setInputFormatClass(TextInputFormat.class);
    	job5.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = inputDiff;
    	output = "output2diff";
    	
    	TextInputFormat.addInputPath(job5, new Path(input));
    	TextOutputFormat.setOutputPath(job5, new Path(output));
        
    	job5.waitForCompletion(true);
        
    	//copy files : generate the diff file
    	ProcessBuilder pb5 = new ProcessBuilder("cat", "output1diff/part-r-00000", "output2diff/part-r-00000");
    	File bias5 = new File("Ncopies.txt");
    	pb5.redirectOutput(Redirect.appendTo(bias5));
    	Process w = pb5.start();
    	
    	w.waitFor();
    	
        /********************
         *  File Management * 
         * ******************/
    	
    	FileSystem fs = FileSystem.get(confList[0]);
    	FileStatus[] fileStatus = fs.listStatus(new Path("."));
    	Path[] paths = FileUtil.stat2Paths(fileStatus);
    	for (Path path : paths){
    		if (!(path.getName().equals("src")) && !(path.getName().equals("bin")) && !(path.getName().equals("Ncopies.txt")) && !(path.getName().equals(".settings")) && !(path.getName().equals("input0")) && !(path.getName().equals("new_state")) && !(path.getName().equals("vec_Weights")) && !(path.getName().equals(".classpath")) && !(path.getName().equals(".project"))){
    			fs.delete(path, true);
    		}
    	}
    	
    }
    
    /****************
     * Bitonic Sort 
     * @throws IOException 
     * @throws IllegalArgumentException 
     * @throws InterruptedException 
     * @throws ClassNotFoundException *
     ****************/
    
    public static void bitonicSort(int NumIter, int Nlength) throws IllegalArgumentException, IOException, ClassNotFoundException, InterruptedException{
    	
    	
        /**************************************
         *  Prepare Input for the BitonicSort * 
         * ************************************/
    	
		Configuration conf0 = new Configuration();
		
    	Job job0_red = Job.getInstance(conf0, "Job26");
    	job0_red.setJarByClass(PF_Hadoop.class);
    	job0_red.setOutputKeyClass(Text.class);
    	job0_red.setOutputValueClass(Text.class);
    	
    	job0_red.setMapperClass(MapNcopies_InputBS.class);
    	job0_red.setMapperClass(MapOldsamples_InputBS.class);
    	job0_red.setReducerClass(Reduce_Input_BS.class);
    	
    	job0_red.setOutputFormatClass(TextOutputFormat.class);
    	
    	String output = "temp";
    	MultipleInputs.addInputPath(job0_red, new Path("Ncopies.txt"), TextInputFormat.class, MapNcopies_InputBS.class);
    	MultipleInputs.addInputPath(job0_red, new Path("new_state/part-m-00000"), TextInputFormat.class, MapOldsamples_InputBS.class);
    	//TextInputFormat.addInputPath(job17, new Path(input));
    	TextOutputFormat.setOutputPath(job0_red, new Path(output));
        
    	job0_red.waitForCompletion(true);
    	
        /********************************************
         *   Phase 1 : Create the bitonic sequence  *
         * ******************************************/
    	
        int val = Nlength/2;
        int chunk = 0;
        int offset = 0;
        
        //total number of main steps
        int Nsteps = (int) NumIter - 1; 
        
        //file management
        int ifile = 0;
    	
        String input = "";
        
    	int phase1maxfile = (int) ((NumIter*NumIter - NumIter)/2);
    	
    	
		//Initialize the configuration
    	Configuration[] confList = new Configuration[phase1maxfile];
    	
    	int k = ifile;
        for (int steps=1;steps<=Nsteps;steps++){
        	for (int iter=steps; iter>=1;iter--){
        		confList[k] = new Configuration();
	    		confList[k].set("iter"  ,   Integer.toString(iter));
	    		confList[k].set("step"  ,   Integer.toString(steps));
	    		confList[k].set("length",   Integer.toString(Nlength));
	    		confList[k].set("chunk" ,   Integer.toString(chunk));
	    		confList[k].set("val"   ,   Integer.toString(val));
        		confList[k].set("offset" ,  Integer.toString(offset));
        		
        		
            	chunk = val;
            	offset = val;
            	val = val/2;
            	k++;
        	}
        }
        
        
    	//Job 1
        for(ifile = 0; ifile<confList.length; ifile++){
        	Job job1 = Job.getInstance(confList[ifile], "Job1");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map1_BS.class);
        	job1.setReducerClass(Reduce1_BS.class);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (ifile == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "temp/part-r-00000";
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "output" + ifile;
        	}
        	
        	// setting the output file
        	output = "output" + (ifile + 1);
        	
        	//TextInputFormat.addInputPath(job1, new Path(args[0]));
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
        }
        
        /******************************************
         *   Phase 2 : Sort the bitonic sequence  *
         * ****************************************/
        
        val = Nlength/2;
        chunk = 0;
        offset = 0;
        
        phase1maxfile = ifile;
        ifile = 1;
        
        //Initialize Configuration
    	Configuration[] conf2List = new Configuration[NumIter];
    	
        for(int i=0;i<NumIter;i++){
        	conf2List[i] = new Configuration();
        	//conf2List[i] = this.getConf();
        	conf2List[i].set("iter", Integer.toString(i+1));
        	conf2List[i].set("length", Integer.toString(Nlength));
        	conf2List[i].set("chunk", Integer.toString(chunk));
        	conf2List[i].set("val", Integer.toString(val));
        	conf2List[i].set("offset", Integer.toString(offset));
        	
        	chunk = val;
        	offset = val;
        	val = val/2;
        }
        
    	//Job 2
    	for (int i=0;i<NumIter;i++){
        	Job job1 = Job.getInstance(conf2List[i], "Job1");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map2_BS.class);
        	job1.setReducerClass(Reduce2_BS.class);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "output" + phase1maxfile;
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "out" + ifile;
        	}
        	
        	// setting the output file
        	output = "out" + (ifile + 1);
        	
        	System.out.println(input);
        	System.out.println(output);
        	
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
        	
        	ifile++;
    	}
    	
        /********************
         *  File Management * 
         * ******************/
    	
    	FileSystem fs = FileSystem.get(conf2List[0]);
    	
		//Rename NewInput to input
		fs.rename(new Path(output), new Path("vec_Particles"));
    	
		FileStatus[] fileStatus = fs.listStatus(new Path("."));
    	Path[] paths = FileUtil.stat2Paths(fileStatus);
    	for (Path path : paths){
    		if (!(path.getName().equals("src")) && !(path.getName().equals("bin")) && !(path.getName().equals(".settings")) && !(path.getName().equals("vec_Particles")) && !(path.getName().equals("input0")) && !(path.getName().equals(".classpath")) && !(path.getName().equals(".project"))){
    			fs.delete(path, true);
    		}
    	}
    	
    }
    
    /***************************
     * Redistribution Function 
     * @param confListred 
     * @throws InterruptedException 
     * @throws IOException 
     * @throws ClassNotFoundException *
     ***************************/
    
    public static void redistribute2(int NumIter, int Nlength, Configuration[][] confListred) throws ClassNotFoundException, IOException, InterruptedException{
        
    	String output = "vec_Particles";
    	
        /*******************************
         *  Execute the Redistribution * 
         * *****************************/
    	
        int maxIter = (int) NumIter; //log2N steps
        
        for (int level=0;level<maxIter;level++){
        
        /**********************
         *   Generate NodeID  *
         * ********************/
        //int iter = 2; // iter = log2(N):-1:0
        int iter = level;
        int adj = (int) (Nlength/Math.pow(2,iter));
        NumIter = (int)(Math.log(adj)/Math.log(2));
        
        //Job 0 - Generate NodeID
    	Job job0 = Job.getInstance(confListred[0][level], "Job0");
    	job0.setJarByClass(PF_Hadoop.class);
    	job0.setOutputKeyClass(Text.class);
    	job0.setOutputValueClass(Text.class);
    	
    	job0.setMapperClass(Map0_red.class);
    	job0.setNumReduceTasks(0);
    	
    	job0.setInputFormatClass(TextInputFormat.class);
    	job0.setOutputFormatClass(TextOutputFormat.class);
    	
    	String input = "vec_Particles/part-r-00000";
    	output = "in0";
    	
    	TextInputFormat.addInputPath(job0, new Path(input));
    	TextOutputFormat.setOutputPath(job0, new Path(output));
        
    	job0.waitForCompletion(true);
        
        /*******************************
         *   cN = [0 cumsum(Ncopies)]  *
         * *****************************/
        
    	String outputemp = output;
    	
    	//Job 1 - Forward : Parallel Summation
    	for (int i=0;i<NumIter;i++){
        	Job job1 = Job.getInstance(confListred[0][i], "Job1");
        	job1.setJarByClass(PF_Hadoop.class);
        	job1.setOutputKeyClass(Text.class);
        	job1.setOutputValueClass(Text.class);
        	
        	job1.setMapperClass(Map1_red.class);
        	job1.setReducerClass(Reduce1_Red.class);
        	
        	job1.setInputFormatClass(TextInputFormat.class);
        	job1.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = outputemp;
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = outputemp + i;
        	}
        	
        	// setting the output file
        	output = outputemp + (i + 1);
        	
        	TextInputFormat.addInputPath(job1, new Path(input));
        	TextOutputFormat.setOutputPath(job1, new Path(output));
            
        	job1.waitForCompletion(true);
    	}
    	
    	int j = (int) NumIter;
    	
    	//Job 2 - Backward : Parallel "Subtract"
    	for (int i=0;i<NumIter;i++){
        	Job job2 = Job.getInstance(confListred[level][i], "Job2");
        	job2.setJarByClass(PF_Hadoop.class);
        	job2.setOutputKeyClass(Text.class);
        	job2.setOutputValueClass(Text.class);
        	
        	job2.setMapperClass(Map2_red.class);
        	job2.setReducerClass(Reduce2_red.class);
        	
        	job2.setInputFormatClass(TextInputFormat.class);
        	job2.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0){
        		MultipleInputs.addInputPath(job2, new Path("in0"+j+"/part-r-00000"), TextInputFormat.class, Map22_red.class);
        		
        		if (j == 1){
        			MultipleInputs.addInputPath(job2, new Path("in0/part-m-00000"), TextInputFormat.class, Map2_red.class);
        		}else{
        			MultipleInputs.addInputPath(job2, new Path("in0"+(j-1)+"/part-r-00000"), TextInputFormat.class, Map2_red.class);
        		}
        	}else{
        		if (i == (NumIter-1)){
        			MultipleInputs.addInputPath(job2, new Path("out"+(j+1)+"/part-r-00000"), TextInputFormat.class, Map22_red.class);
        			MultipleInputs.addInputPath(job2, new Path("in0/part-m-00000"), TextInputFormat.class, Map2_red.class);
        		}else{
        			MultipleInputs.addInputPath(job2, new Path("out"+(j+1)+"/part-r-00000"), TextInputFormat.class, Map22_red.class);
        			MultipleInputs.addInputPath(job2, new Path("in0"+(j-1)+"/part-r-00000"), TextInputFormat.class, Map2_red.class);
        		}
        	}
        	
        	output = "out"+ Integer.toString(j);
        	
        	TextOutputFormat.setOutputPath(job2, new Path(output));
            
        	job2.waitForCompletion(true);
        	
        	j--;
    	}
    	
        /*************************************************************
         *   Add zeros so that we have the  cN = [0 cumsum(Ncopies)] *
         * ***********************************************************/
        
        //Job 0 - Generate NodeID);		
    	Job job24 = Job.getInstance(confListred[level][0], "Job24");
    	job24.setJarByClass(PF_Hadoop.class);
    	job24.setOutputKeyClass(Text.class);
    	job24.setOutputValueClass(Text.class);
    	
    	job24.setMapperClass(Map24_red.class);
    	job24.setReducerClass(Reduce24_red.class);
    	
    	job24.setInputFormatClass(TextInputFormat.class);
    	job24.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = output;
    	output = "zeros";
    	
    	TextInputFormat.addInputPath(job24, new Path(input));
    	TextOutputFormat.setOutputPath(job24, new Path(output));
        
    	job24.waitForCompletion(true);
        
        /********************************************************************
         *   Create the cN.txt which contains the  cN = [0 cumsum(Ncopies)] *
         * ******************************************************************/
        
    	//add in the vector the value of the 1st key equal to zero.
    	ProcessBuilder pb5r = new ProcessBuilder("cat", "out1/part-r-00000", "zeros/part-r-00000");
    	File bias5r = new File("cN.txt");
    	pb5r.redirectOutput(Redirect.appendTo(bias5r));
    	pb5r.start();
    	
        /******************************
         *   Retrieve the N = cN(end) *
         * ****************************/
        
    	//It is retrieved from the last file of the job 1 (parallel summation)
    	//This file is the following : "in0"+NumIter+"/part-r-00000"
    	//or N = adj/2
    	ProcessBuilder pbr = new ProcessBuilder("cat", "","in0"+(int)NumIter+"/part-r-00000");
    	File biasr = new File("N.txt");
    	pbr.redirectOutput(Redirect.appendTo(biasr));
    	pbr.start();
    	
        /**********************
         *   Compute the mindif *
         * ********************/
        
        //Job 3 - compute mindif
    	Job job3r = Job.getInstance(confListred[level][0], "Job3");
    	job3r.setJarByClass(PF_Hadoop.class);
    	job3r.setOutputKeyClass(Text.class);
    	job3r.setOutputValueClass(Text.class);
    	
    	job3r.setMapperClass(Map3_red.class);
    	job3r.setNumReduceTasks(0);
    	
    	job3r.setInputFormatClass(TextInputFormat.class);
    	job3r.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "cN.txt";
    	output = "mindif";
    	
    	TextInputFormat.addInputPath(job3r, new Path(input));
    	TextOutputFormat.setOutputPath(job3r, new Path(output));
        
    	job3r.waitForCompletion(true);
        
    	
        /**********************************
         *   Compute the Nfirst via diff  *
         * ********************************/
    	
    	String inputDiff = "mindif/part-m-00000";
    	
        //Job 4 - odd numbers of diff
    	Job job4r = Job.getInstance(confListred[level][0], "Job4");
    	job4r.setJarByClass(PF_Hadoop.class);
    	job4r.setOutputKeyClass(Text.class);
    	job4r.setOutputValueClass(Text.class);
    	
    	job4r.setMapperClass(Map4_red.class);
    	job4r.setReducerClass(Reduce4_red.class);
    	
    	job4r.setInputFormatClass(TextInputFormat.class);
    	job4r.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = inputDiff;
    	output = "output1diff";
    	
    	TextInputFormat.addInputPath(job4r, new Path(input));
    	TextOutputFormat.setOutputPath(job4r, new Path(output));
        
    	job4r.waitForCompletion(true);
        
        //Job 5 - even numbers of diff
    	Job job5r = Job.getInstance(confListred[level][0], "Job5");
    	job5r.setJarByClass(PF_Hadoop.class);
    	job5r.setOutputKeyClass(Text.class);
    	job5r.setOutputValueClass(Text.class);
    	
    	job5r.setMapperClass(Map5_red.class);
    	job5r.setReducerClass(Reduce5_red.class);
    	
    	job5r.setInputFormatClass(TextInputFormat.class);
    	job5r.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = inputDiff;
    	output = "output2diff";
    	
    	TextInputFormat.addInputPath(job5r, new Path(input));
    	TextOutputFormat.setOutputPath(job5r, new Path(output));
        
    	job5r.waitForCompletion(true);
        
    	//copy files : generate the Nfirst file
    	ProcessBuilder pb6 = new ProcessBuilder("cat", "output1diff/part-r-00000", "output2diff/part-r-00000");
    	File bias6 = new File("Nfirst.txt");
    	pb6.redirectOutput(Redirect.appendTo(bias6));
    	pb6.start();
    	
        /*********************************
         *   compute oldsamples_first    *
         * *******************************/
    	
        //Job 6
    	Job job6 = Job.getInstance(confListred[level][0], "Job6");
    	job6.setJarByClass(PF_Hadoop.class);
    	job6.setOutputKeyClass(Text.class);
    	job6.setOutputValueClass(Text.class);
    	
    	job6.setMapperClass(Map6_red.class);
    	job6.setNumReduceTasks(0);
    	
    	job6.setInputFormatClass(TextInputFormat.class);
    	job6.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "in0/part-m-00000";
    	output = "oldsamples_first";
    	
    	TextInputFormat.addInputPath(job6, new Path(input));
    	TextOutputFormat.setOutputPath(job6, new Path(output));
        
    	job6.waitForCompletion(true);
        
    	
        /************************
         *   Compute the maxdif *
         * **********************/
        
        //Job 3 - compute mindif
    	Job job7 = Job.getInstance(confListred[level][0], "Job7");
    	job7.setJarByClass(PF_Hadoop.class);
    	job7.setOutputKeyClass(Text.class);
    	job7.setOutputValueClass(Text.class);
    	
    	job7.setMapperClass(Map7_red.class);
    	job7.setNumReduceTasks(0);
    	
    	job7.setInputFormatClass(TextInputFormat.class);
    	job7.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "cN.txt";
    	output = "maxdif";
    	
    	TextInputFormat.addInputPath(job7, new Path(input));
    	TextOutputFormat.setOutputPath(job7, new Path(output));
        
    	job7.waitForCompletion(true);
        
        /***********************
         *   Compute the Nraw  *
         * *********************/
    	
    	inputDiff = "maxdif/part-m-00000";
    	
        //Job 4 - odd numbers of diff
    	Job job8 = Job.getInstance(confListred[level][0], "Job8");
    	job8.setJarByClass(PF_Hadoop.class);
    	job8.setOutputKeyClass(Text.class);
    	job8.setOutputValueClass(Text.class);
    	
    	job8.setMapperClass(Map4_red.class);
    	job8.setReducerClass(Reduce4_red.class);
    	
    	job8.setInputFormatClass(TextInputFormat.class);
    	job8.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = inputDiff;
    	output = "output11diff";
    	
    	TextInputFormat.addInputPath(job8, new Path(input));
    	TextOutputFormat.setOutputPath(job8, new Path(output));
        
    	job8.waitForCompletion(true);
        
        //Job 5 - even numbers of diff
    	Job job9 = Job.getInstance(confListred[level][0], "Job9");
    	job9.setJarByClass(PF_Hadoop.class);
    	job9.setOutputKeyClass(Text.class);
    	job9.setOutputValueClass(Text.class);
    	
    	job9.setMapperClass(Map5_red.class);
    	job9.setReducerClass(Reduce5_red.class);
    	
    	job9.setInputFormatClass(TextInputFormat.class);
    	job9.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = inputDiff;
    	output = "output22diff";
    	
    	TextInputFormat.addInputPath(job9, new Path(input));
    	TextOutputFormat.setOutputPath(job9, new Path(output));
        
    	job9.waitForCompletion(true);
        
    	//copy files : generate the Nraw file
    	ProcessBuilder pb7 = new ProcessBuilder("cat", "output11diff/part-r-00000", "output22diff/part-r-00000");
    	File bias7 = new File("Nraw.txt");
    	pb7.redirectOutput(Redirect.appendTo(bias7));
    	pb7.start();
    	
        /*******************
         *   compute slef  *
         * *****************/
    	
        //Job 10 - compute slef = cN(2:(N/2+1))<=N/2
    	Job job10 = Job.getInstance(confListred[level][0], "Job10");
    	job10.setJarByClass(PF_Hadoop.class);
    	job10.setOutputKeyClass(Text.class);
    	job10.setOutputValueClass(Text.class);
    	
    	job10.setMapperClass(Map8_red.class);
    	job10.setNumReduceTasks(0);
    	
    	job10.setInputFormatClass(TextInputFormat.class);
    	job10.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "cN.txt";
    	output = "slef";
    	
    	TextInputFormat.addInputPath(job10, new Path(input));
    	TextOutputFormat.setOutputPath(job10, new Path(output));
        
    	job10.waitForCompletion(true);
    	
        /*********************************
         *   compute nleft = sum(slef)  *
         * *******************************/
    	
        double maxiter = Math.log(adj)/Math.log(2);
    	
    	//Job 11 - compute nleft = sum(sleft)
    	for (int i=0;i<maxiter-1;i++){
        	Job job11 = Job.getInstance(confListred[level][0], "Job11");
        	job11.setJarByClass(PF_Hadoop.class);
        	job11.setOutputKeyClass(Text.class);
        	job11.setOutputValueClass(Text.class);
        	
        	job11.setMapperClass(Map9_red.class);
        	job11.setReducerClass(Reduce9_red.class);
        	
        	job11.setInputFormatClass(TextInputFormat.class);
        	job11.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "slef/part-m-00000";
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "slef" + i ;
        	}
        	
        	// setting the output file
        	output = "slef" + (i + 1);
        	
        	TextInputFormat.addInputPath(job11, new Path(input));
        	TextOutputFormat.setOutputPath(job11, new Path(output));
            
        	job11.waitForCompletion(true);
    	}
    	
        //retrieve nleft = sum(Ncopies)
    	ProcessBuilder pb1 = new ProcessBuilder("cat", "","slef"+(int)(maxiter-1)+"/part-r-00000");
    	File bias1 = new File("nleft.txt");
    	pb1.redirectOutput(Redirect.appendTo(bias1));
    	pb1.start();
        
        
        /*******************
         *   compute nrig  *
         * *****************/
        
        //Job 12 - compute nrig = ((N/2+1):N)>0;
    	Job job12 = Job.getInstance(confListred[level][0], "Job12");
    	job12.setJarByClass(PF_Hadoop.class);
    	job12.setOutputKeyClass(Text.class);
    	job12.setOutputValueClass(Text.class);
    	
    	job12.setMapperClass(Map10_red.class);
    	job12.setNumReduceTasks(0);
    	
    	job12.setInputFormatClass(TextInputFormat.class);
    	job12.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "in0/part-m-00000";
    	output = "nrig";
    	
    	TextInputFormat.addInputPath(job12, new Path(input));
    	TextOutputFormat.setOutputPath(job12, new Path(output));
        
    	job12.waitForCompletion(true);
    	
        
        /*********************************
         *   compute nright = sum(nrig)  *
         * *******************************/
    	
    	//Job 13 - compute nright = sum(nrig)
    	for (int i=0;i<maxiter-1;i++){
        	Job job13 = Job.getInstance(confListred[level][0], "Job13");
        	job13.setJarByClass(PF_Hadoop.class);
        	job13.setOutputKeyClass(Text.class);
        	job13.setOutputValueClass(Text.class);
        	
        	job13.setMapperClass(Map9_1_red.class);
        	job13.setReducerClass(Reduce9_red.class);
        	
        	job13.setInputFormatClass(TextInputFormat.class);
        	job13.setOutputFormatClass(TextOutputFormat.class);
        	
        	if (i == 0) {
        		// for the first iteration the input will be the first input argument
        		input = "nrig/part-m-00000";
        	}else{
        		// for the remaining iterations, the input will be the output of the previous iteration
        		input = "nrig" + i ;
        	}
        	
        	// setting the output file
        	output = "nrig" + (i + 1);
        	
        	TextInputFormat.addInputPath(job13, new Path(input));
        	TextOutputFormat.setOutputPath(job13, new Path(output));
            
        	job13.waitForCompletion(true);
    	}
    	
        //retrieve nright = sum(nrig)
    	ProcessBuilder pb2 = new ProcessBuilder("cat", "","nrig"+(int) (maxiter-1)+"/part-r-00000");
    	File bias2 = new File("nright.txt");
    	pb2.redirectOutput(Redirect.appendTo(bias2));
    	pb2.start();
        
        /**********************************************************
         *  create the nlr file which is equal with nleft-nright  *
         * ********************************************************/
    	
        //Job 26
    	Job job26 = Job.getInstance(confListred[level][0], "Job26");
    	job26.setJarByClass(PF_Hadoop.class);
    	job26.setOutputKeyClass(Text.class);
    	job26.setOutputValueClass(Text.class);
    	
    	job26.setMapperClass(Map9_nleft_red.class);
    	job26.setMapperClass(Map9_nright_red.class);
    	job26.setReducerClass(Reduce_nlr_red.class);
    	
    	//job17.setInputFormatClass(TextInputFormat.class);
    	job26.setOutputFormatClass(TextOutputFormat.class);
    	
    	//input = "oldinput.txt";
    	output = "nlr";
    	MultipleInputs.addInputPath(job26, new Path("nleft.txt"), TextInputFormat.class, Map9_nleft_red.class);
    	MultipleInputs.addInputPath(job26, new Path("nright.txt"), TextInputFormat.class, Map9_nright_red.class);
    	//TextInputFormat.addInputPath(job17, new Path(input));
    	TextOutputFormat.setOutputPath(job26, new Path(output));
        
    	job26.waitForCompletion(true);
    	
        /*******************************************************
         *  input00 = add nlr as the last column of the in0 *
         *            (i.e. combine nlr with in0)           *
         * *****************************************************/
    	
        //Job 27
    	Job job30 = Job.getInstance(confListred[level][0], "Job30");
    	job30.setJarByClass(PF_Hadoop.class);
    	job30.setOutputKeyClass(Text.class);
    	job30.setOutputValueClass(Text.class);
    	
    	job30.setMapperClass(Map9_in0_red.class);
    	job30.setMapperClass(Map9_nlr_red.class);
    	job30.setReducerClass(Reduce_input00_red.class);
    	
    	//job17.setInputFormatClass(TextInputFormat.class);
    	job30.setOutputFormatClass(TextOutputFormat.class);
    	
    	//input = "oldinput.txt";
    	output = "input00";
    	MultipleInputs.addInputPath(job30, new Path("in0/part-m-00000"), TextInputFormat.class, Map9_in0_red.class);
    	MultipleInputs.addInputPath(job30, new Path("nlr/part-r-00000"), TextInputFormat.class, Map9_nlr_red.class);
    	//TextInputFormat.addInputPath(job17, new Path(input));
    	TextOutputFormat.setOutputPath(job30, new Path(output));
        
    	job30.waitForCompletion(true);
    	
        /***********************************************************
         *   compute index = mod((nleft-nright)+(1:N/2)-1,N/2)+1;  *
         *   and	 olpdsamples2 = oldsamples(index);             *
         * *********************************************************/
    	
        //Job 14 - compute nrig = ((N/2+1):N)>0;
    	Job job14 = Job.getInstance(confListred[level][0], "Job14");
    	job14.setJarByClass(PF_Hadoop.class);
    	job14.setOutputKeyClass(Text.class);
    	job14.setOutputValueClass(Text.class);
    	
    	job14.setMapperClass(Map11_red.class);
    	job14.setNumReduceTasks(0);
    	
    	job14.setInputFormatClass(TextInputFormat.class);
    	job14.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "input00/part-r-00000";
    	//output = "oldsamples2_temp";
    	output = "oldsamples2";
    	
    	TextInputFormat.addInputPath(job14, new Path(input));
    	TextOutputFormat.setOutputPath(job14, new Path(output));
        
    	job14.waitForCompletion(true);
    	
    	//Job oldsamples2 - set the proper values to each key
    	Job job27 = Job.getInstance(confListred[level][0], "Job27");
    	job27.setJarByClass(PF_Hadoop.class);
    	job27.setOutputKeyClass(Text.class);
    	job27.setOutputValueClass(Text.class);
    	
    	job27.setMapperClass(Map11_1_red.class);
    	job27.setMapperClass(Map11_2_red.class);
    	job27.setReducerClass(Reduce_oldsamples2_old2_red.class);
    	
    	job27.setOutputFormatClass(TextOutputFormat.class);
    	
    	//input = "oldinput.txt";
    	output = "oldsamples2_old2";
    	MultipleInputs.addInputPath(job27, new Path("input00/part-r-00000"), TextInputFormat.class, Map11_1_red.class);
    	MultipleInputs.addInputPath(job27, new Path("oldsamples2/part-m-00000"), TextInputFormat.class, Map11_2_red.class);
    	//TextInputFormat.addInputPath(job17, new Path(input));
    	TextOutputFormat.setOutputPath(job27, new Path(output));
    	
    	job27.waitForCompletion(true);
    	
        /*****************************************
         *   compute fin = cN(2:(N/2+1))>N/2     *
         * ***************************************/
    	
        //Job 10 - compute slef = cN(2:(N/2+1))<=N/2
    	Job job15 = Job.getInstance(confListred[level][0], "Job15");
    	job15.setJarByClass(PF_Hadoop.class);
    	job15.setOutputKeyClass(Text.class);
    	job15.setOutputValueClass(Text.class);
    	
    	job15.setMapperClass(Map12_red.class);
    	job15.setNumReduceTasks(0);
		
    	job15.setInputFormatClass(TextInputFormat.class);
    	job15.setOutputFormatClass(TextOutputFormat.class);
    	
    	input = "cN.txt";
    	output = "fin";
    	
    	TextInputFormat.addInputPath(job15, new Path(input));
    	TextOutputFormat.setOutputPath(job15, new Path(output));
        
    	job15.waitForCompletion(true);
    	
    	
        /******************************
         *   compute f = fin(index)     *
         * ****************************/
    	
    	//Job 27
    	Job job16 = Job.getInstance(confListred[level][0], "Job16");
    	job16.setJarByClass(PF_Hadoop.class);
    	job16.setOutputKeyClass(Text.class);
    	job16.setOutputValueClass(Text.class);
    	
    	job16.setMapperClass(Map19_fin_red.class);
    	job16.setMapperClass(Map19_oldsamples2_red.class);
    	job16.setReducerClass(Reduce_f_red.class);
    	
    	job16.setOutputFormatClass(TextOutputFormat.class);
    	
    	//input = "oldinput.txt";
    	output = "f";
    	MultipleInputs.addInputPath(job16, new Path("fin/part-m-00000"), TextInputFormat.class, Map19_fin_red.class);
    	MultipleInputs.addInputPath(job16, new Path("oldsamples2/part-m-00000"), TextInputFormat.class, Map19_oldsamples2_red.class);
    	TextOutputFormat.setOutputPath(job16, new Path(output));
    	        
    	job16.waitForCompletion(true);
    	
    	
        /**************************************************
         *   compute old1 = ~f.*oldsamples(:,(N/2+1):N)  * 
         * ************************************************/
    	
        //Job 17 - vector vector multiplication
    	Job job17 = Job.getInstance(confListred[level][0], "Job17");
    	job17.setJarByClass(PF_Hadoop.class);
    	job17.setOutputKeyClass(Text.class);
    	job17.setOutputValueClass(Text.class);
    	
    	job17.setMapperClass(Map13_red.class);
    	job17.setMapperClass(Map14_red.class);
    	job17.setReducerClass(Reduce13_red.class);
    	
    	job17.setOutputFormatClass(TextOutputFormat.class);
    	
    	output = "old1";
    	MultipleInputs.addInputPath(job17, new Path("f/part-r-00000"), TextInputFormat.class, Map13_red.class);
    	MultipleInputs.addInputPath(job17, new Path("input00/part-r-00000"), TextInputFormat.class, Map14_red.class);
		
    	TextOutputFormat.setOutputPath(job17, new Path(output));
        
    	job17.waitForCompletion(true);
    	
        /************************************
         *   compute old2 = f.*oldsamples2  * 
         * **********************************/
    	
        //Job 18 
    	Job job18 = Job.getInstance(confListred[level][0], "Job18");
    	job18.setJarByClass(PF_Hadoop.class);
    	job18.setOutputKeyClass(Text.class);
    	job18.setOutputValueClass(Text.class);
    	
    	job18.setMapperClass(Map15_red.class); //f
    	job18.setMapperClass(Map16_oldsamples2_old2_red.class); //oldsamples2
    	job18.setReducerClass(Reduce14_red.class);
    	
    	job18.setOutputFormatClass(TextOutputFormat.class);
    	
    	//input = "oldinput.txt";
    	output = "old2";
    	MultipleInputs.addInputPath(job18, new Path("f/part-r-00000"), TextInputFormat.class, Map15_red.class);
    	MultipleInputs.addInputPath(job18, new Path("oldsamples2_old2/part-r-00000"), TextInputFormat.class, Map16_oldsamples2_old2_red.class);
    	TextOutputFormat.setOutputPath(job18, new Path(output));
        
    	job18.waitForCompletion(true);
    	
        /********************************************
         *   compute oldsamples_second = old1+old2  * 
         * ******************************************/
    	
        //Job 19
    	Job job19 = Job.getInstance(confListred[level][0], "Job19");
    	job19.setJarByClass(PF_Hadoop.class);
    	job19.setOutputKeyClass(Text.class);
    	job19.setOutputValueClass(Text.class);
    	
    	job19.setMapperClass(Map17_red.class);
    	job19.setMapperClass(Map18_red.class);
    	job19.setReducerClass(Reduce15_red.class);
    	
    	job19.setOutputFormatClass(TextOutputFormat.class);
    	
    	//input = "oldinput.txt";
    	output = "oldsamples_second";
    	MultipleInputs.addInputPath(job19, new Path("old1/part-r-00000"), TextInputFormat.class, Map17_red.class);
    	MultipleInputs.addInputPath(job19, new Path("old2/part-r-00000"), TextInputFormat.class, Map18_red.class);
		
    	TextOutputFormat.setOutputPath(job19, new Path(output));
        
    	job19.waitForCompletion(true);
    	
        /**************************************************
         *   compute N1 = ~f.*Ncopies((N/2+1):N)  * 
         * ************************************************/
    	
        //Job 17 - vector vector multiplication
    	Job job20 = Job.getInstance(confListred[level][0], "Job20");
    	job20.setJarByClass(PF_Hadoop.class);
    	job20.setOutputKeyClass(Text.class);
    	job20.setOutputValueClass(Text.class);
    	
    	job20.setMapperClass(Map13_red.class); //f
    	job20.setMapperClass(Map14_ncopies_red.class); //ncopies
    	job20.setReducerClass(Reduce13_red.class);
    	
    	job20.setOutputFormatClass(TextOutputFormat.class);
    	
    	output = "N1";
    	MultipleInputs.addInputPath(job20, new Path("f/part-r-00000"), TextInputFormat.class, Map13_red.class);
    	MultipleInputs.addInputPath(job20, new Path("input00/part-r-00000"), TextInputFormat.class, Map14_ncopies_red.class);
		
    	TextOutputFormat.setOutputPath(job20, new Path(output));
        
    	job20.waitForCompletion(true);
    	
        /**************************************
         *   compute NrawIndex = Nraw(index)  *
         * ************************************/
    	
    	//Job 27
    	Job job23 = Job.getInstance(confListred[level][0], "Job23");
    	job23.setJarByClass(PF_Hadoop.class);
    	job23.setOutputKeyClass(Text.class);
    	job23.setOutputValueClass(Text.class);
    	
    	job23.setMapperClass(Map19_nraw_red.class);
    	job23.setMapperClass(Map19_oldsamples2_red.class);
    	job23.setReducerClass(Reduce_f_red.class);
    	
    	job23.setOutputFormatClass(TextOutputFormat.class);
    	
    	output = "NrawIndex";
    	MultipleInputs.addInputPath(job23, new Path("Nraw.txt"), TextInputFormat.class, Map19_nraw_red.class);
    	MultipleInputs.addInputPath(job23, new Path("oldsamples2/part-m-00000"), TextInputFormat.class, Map19_oldsamples2_red.class);
    	TextOutputFormat.setOutputPath(job23, new Path(output));
    	
    	job23.waitForCompletion(true);
    	
        /************************************
         *   compute N2 = f.*NrawIndex  * 
         * **********************************/
    	
        //Job 18 
    	Job job21 = Job.getInstance(confListred[level][0], "Job21");
    	job21.setJarByClass(PF_Hadoop.class);
    	job21.setOutputKeyClass(Text.class);
    	job21.setOutputValueClass(Text.class);
    	
    	job21.setMapperClass(Map15_red.class);
    	job21.setMapperClass(Map16_nrawindex_red.class);
    	job21.setReducerClass(Reduce14_red.class);
    	
		
    	job21.setOutputFormatClass(TextOutputFormat.class);
		
		
    	output = "N2";
    	MultipleInputs.addInputPath(job21, new Path("f/part-r-00000"), TextInputFormat.class, Map15_red.class);
    	MultipleInputs.addInputPath(job21, new Path("NrawIndex/part-r-00000"), TextInputFormat.class, Map16_nrawindex_red.class);
		
    	TextOutputFormat.setOutputPath(job21, new Path(output));
        
    	job21.waitForCompletion(true);
    	
        /******************************
         *   compute Nsecond = N1+N2  * 
         * ****************************/
    	
        //Job 19
    	Job job22 = Job.getInstance(confListred[level][0], "Job22");
    	job22.setJarByClass(PF_Hadoop.class);
    	job22.setOutputKeyClass(Text.class);
    	job22.setOutputValueClass(Text.class);
    	
    	job22.setMapperClass(Map17_red.class);
    	job22.setMapperClass(Map18_red.class);
    	job22.setReducerClass(Reduce15_Nsecond_red.class);
    	
		
    	job22.setOutputFormatClass(TextOutputFormat.class);
    	
		
    	output = "Nsecond";
    	MultipleInputs.addInputPath(job22, new Path("N1/part-r-00000"), TextInputFormat.class, Map17_red.class);
    	MultipleInputs.addInputPath(job22, new Path("N2/part-r-00000"), TextInputFormat.class, Map18_red.class);
		
    	TextOutputFormat.setOutputPath(job22, new Path(output));
        
    	job22.waitForCompletion(true);
    	
    	
        /***************************************************************************************
         *  Merge the Nfirst, Nsecond, oldsamples_first, oldsamples_second for the next level  *
         *  This is used as an input for the next level of the tree.                           * 
         * *************************************************************************************/
    	
    	//copy files : generate the New Ncopies file
    	ProcessBuilder pb8 = new ProcessBuilder("cat", "Nfirst.txt", "Nsecond/part-r-00000");
    	File bias8 = new File("NewNcopies.txt");
    	pb8.redirectOutput(Redirect.appendTo(bias8));
    	pb8.start();
    	
    	//copy files : generate the New oldsamples file
    	ProcessBuilder pb9 = new ProcessBuilder("cat", "oldsamples_first/part-m-00000", "oldsamples_second/part-r-00000");
    	File bias9 = new File("NewOldsamples.txt");
    	pb9.redirectOutput(Redirect.appendTo(bias9));
    	pb9.start();
    	
    	//Generate new input file for the next level
    	Job job25 = Job.getInstance(confListred[level][0], "Job25");
    	job25.setJarByClass(PF_Hadoop.class);
    	job25.setOutputKeyClass(Text.class);
    	job25.setOutputValueClass(Text.class);
    	
    	job25.setMapperClass(Map25_NewNcopies_red.class);
    	job25.setMapperClass(Map25_NewOldsamples_red.class);
    	job25.setReducerClass(Reduce25_NewInput_red.class);
    	
    	//job17.setInputFormatClass(TextInputFormat.class);
    	job25.setOutputFormatClass(TextOutputFormat.class);
    	
    	//input = "oldinput.txt";
    	output = "NewInput";
    	MultipleInputs.addInputPath(job25, new Path("NewNcopies.txt"), TextInputFormat.class, Map25_NewNcopies_red.class);
    	MultipleInputs.addInputPath(job25, new Path("NewOldsamples.txt"), TextInputFormat.class, Map25_NewOldsamples_red.class);
    	
    	TextOutputFormat.setOutputPath(job25, new Path(output));
    	
    	job25.waitForCompletion(true);
    	
        /********************
         *  File Management * 
         * ******************/
    	
    	FileSystem fs = FileSystem.get(confListred[level][0]);
    	FileStatus[] fileStatus = fs.listStatus(new Path("."));
    	Path[] paths = FileUtil.stat2Paths(fileStatus);
    	for (Path path : paths){
    		if (!(path.getName().equals("src")) && !(path.getName().equals("bin")) && !(path.getName().equals(".settings")) && !(path.getName().equals("NewInput")) && !(path.getName().equals("input0")) && !(path.getName().equals("new_state")) && !(path.getName().equals("Ncopies")) && !(path.getName().equals(".classpath")) && !(path.getName().equals(".project"))){
    			fs.delete(path, true);
    		}
    	}
    	
		//Rename NewInput to input
		fs.rename(new Path("NewInput"), new Path("vec_Particles"));
		
        }//for_loop_end
    }
    
    /*********************************************************
     * Estimate the mean value of the posterior distribution 
     * @param confList 
     * @return 
     * @throws IOException 
     * @throws InterruptedException 
     * @throws ClassNotFoundException *
     *********************************************************/
    
    public static float xestm(int Nlength, boolean flag, Configuration[] confList) throws IOException, ClassNotFoundException, InterruptedException{
    	
    	String input = "";
        String output = "";
    	String tempout = output;
    	tempout = "vec_Particles";
    	float xest = 0.0f;
    	
    	if (flag == true){
	    	for (int i=0;i<confList.length;i++){
	        	Job job1 = Job.getInstance(confList[i], "Job1");
	        	job1.setJarByClass(PF_Hadoop.class);
	        	job1.setOutputKeyClass(Text.class);
	        	job1.setOutputValueClass(Text.class);
	        	
	        	job1.setMapperClass(Map1xestm.class);
	        	job1.setReducerClass(Reduce1xestm.class);
	        	
	        	job1.setInputFormatClass(TextInputFormat.class);
	        	job1.setOutputFormatClass(TextOutputFormat.class);
	        	
	        	if (i == 0) {
	        		// for the first iteration the input will be the first input argument
	        		input = tempout;
	        	}else{
	        		// for the remaining iterations, the input will be the output of the previous iteration
	        		input = tempout + i;
	        	}
	        	
	        	// setting the output file
	        	output = tempout + (i + 1);
	        	
	        	TextInputFormat.addInputPath(job1, new Path(input));
	        	TextOutputFormat.setOutputPath(job1, new Path(output));
	            
	        	job1.waitForCompletion(true);
	    	}
	    	
			/****************************
			 * Retrieve and return value *
			 ****************************/
			
	        try {
				BufferedReader br = new BufferedReader(new FileReader(output+"/part-r-00000"));
				
				String line = null;
	    		while ((line = br.readLine()) != null) {
	    			String[] val = line.split(",");
	            	xest = Float.parseFloat(val[2]);
	    		}
	    		
	    		br.close();
	        } catch (IOException e) {
	            System.out.println("File Read Error");
	        }
    	}else{
	    	for (int i=0;i<confList.length;i++){
	        	Job job1 = Job.getInstance(confList[i], "Job1");
	        	job1.setJarByClass(PF_Hadoop.class);
	        	job1.setOutputKeyClass(Text.class);
	        	job1.setOutputValueClass(Text.class);
	        	
	        	job1.setMapperClass(Map1xestmF.class);
	        	job1.setReducerClass(Reduce1xestmF.class);
	        	
	        	job1.setInputFormatClass(TextInputFormat.class);
	        	job1.setOutputFormatClass(TextOutputFormat.class);
	        	
	        	if (i == 0) {
	        		// for the first iteration the input will be the first input argument
	        		input = tempout;
	        	}else{
	        		// for the remaining iterations, the input will be the output of the previous iteration
	        		input = tempout + i;
	        	}
	        	
	        	// setting the output file
	        	output = tempout + (i + 1);
	        	
	        	TextInputFormat.addInputPath(job1, new Path(input));
	        	TextOutputFormat.setOutputPath(job1, new Path(output));
	            
	        	job1.waitForCompletion(true);
	    	}
	    	
	    	
	    	
			/****************************
			 * Retrieve and return value *
			 ****************************/
			
	        try {
				BufferedReader br = new BufferedReader(new FileReader(output+"/part-r-00000"));
				
				String line = null;
	    		while ((line = br.readLine()) != null) {
	    			String[] val = line.split(",");
	            	xest = Float.parseFloat(val[1]);
	    		}
	    		
	    		br.close();
	        } catch (IOException e) {
	            System.out.println("File Read Error");
	        }
	        
    	}
        
        /********************
         *  File Management * 
         * ******************/
        
    	FileSystem fs = FileSystem.get(confList[0]);
    	FileStatus[] fileStatus = fs.listStatus(new Path("."));
    	Path[] paths = FileUtil.stat2Paths(fileStatus);
    	for (Path path : paths){
    		if (!(path.getName().equals("src")) && !(path.getName().equals("bin")) && !(path.getName().equals(".settings")) && !(path.getName().equals("input0")) && !(path.getName().equals("vec_Particles")) && !(path.getName().equals(".classpath")) && !(path.getName().equals(".project"))){
    			fs.delete(path, true);
    		}
    	}
        
		//estimated value
        return xest;
		
    }//end of xest function
}