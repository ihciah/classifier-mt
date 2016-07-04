#include <iostream>
#include <vector>
#include <chrono>

#include "ThreadPool.h"

using namespace std;

int main()
{

  string model_file   = "/home/ch/c4/4layer.prototxt";
  string trained_file = "/home/ch/c4/img.caffemodel";
  string mean_file    = "/home/ch/c4/mean.binaryproto";

  ThreadPool pool(4, model_file, trained_file, mean_file);
  std::string result1, result2;

  result1 = pool.getres("/home/ch/c4/images/2_8374.bmp");
  result2 = pool.getres("/home/ch/c4/images/3_0858.bmp");

  std::cout << result1 << ' ' << result2;
  std::cout << std::endl;
  return 0;
}
