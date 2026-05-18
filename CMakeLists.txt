cmake_minimum_required(VERSION 3.10)

project(StochasticFinanceModels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/cpp)

add_executable(monte_carlo_engine src/cpp/monte_carlo_engine.cpp)