cmake_minimum_required(VERSION 3.20)
project(quasar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Apple Accelerate (dense complex GEMM path) ----
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# ---- Core static library ----
file(GLOB QUASAR_CORE_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(quasar_core STATIC ${QUASAR_CORE_SOURCES})
target_include_directories(quasar_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(quasar_core PUBLIC ${ACCELERATE_FRAMEWORK})
# Accelerate's new BLAS interface header guard.
target_compile_definitions(quasar_core PUBLIC ACCELERATE_NEW_LAPACK)

# ---- CLI executable ----
add_executable(quasar ${CMAKE_SOURCE_DIR}/apps/main.cpp)
target_link_libraries(quasar PRIVATE quasar_core)

# ---- Tests ----
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB QUASAR_TEST_SOURCES ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(quasar_tests ${QUASAR_TEST_SOURCES})
target_link_libraries(quasar_tests PRIVATE quasar_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(quasar_tests)
