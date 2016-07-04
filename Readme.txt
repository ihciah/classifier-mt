Add this to example/CMakeLists.txt

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

If you are using Visual Studio, you should turn on C++ 11 support(maybe pthread).