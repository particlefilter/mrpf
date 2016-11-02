/***********************************************************
*            Generate Input File                           *
* Details : https://github.com/particlefilter/mrpf/        *
* Copyright(c)2016 University of Liverpool                 *
************************************************************/

#include <iostream>
#include <fstream>
#include <math.h> 
#include <cstdlib>

using namespace std;

int main () {
  //open file
  ofstream myfile;
  myfile.open ("in2_10.txt");
  
  int N = pow(2,10);
  //write dataset in form int -> float or key -> value
  for(int i=1;i<=N;i++){
                float r = 0.5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.99-0.5)));
                float n = roundf(r * 100) / 100;
                myfile << i << "," << n << endl;
  }
  
  //close file
  myfile.close();
  return 0;
}
