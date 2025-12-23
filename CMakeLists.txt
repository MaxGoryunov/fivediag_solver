cmake_minimum_required(VERSION 3.14)

project(Project C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

# Set common compiler options for all targets
set(COMMON_COMPILE_OPTIONS
    # -Wall
    # -Wextra
    # -Wpedantic
    # -Wshadow
    # -Wformat=2
    # -Wconversion
)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")

add_executable(main_executable src/main.c)
target_link_libraries(main_executable PRIVATE m)
target_compile_options(main_executable PRIVATE ${COMMON_COMPILE_OPTIONS})

find_package(OpenMP REQUIRED)
add_executable(main_executable_openmp src/openmp_main.c)
target_link_libraries(main_executable_openmp PRIVATE OpenMP::OpenMP_C m)
target_compile_options(main_executable_openmp PRIVATE ${COMMON_COMPILE_OPTIONS})

find_package(MPI REQUIRED)
add_executable(main_executable_mpi src/mpi_main.c)
target_link_libraries(main_executable_mpi PRIVATE MPI::MPI_C -lm)
target_compile_options(main_executable_mpi PRIVATE ${COMMON_COMPILE_OPTIONS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

add_executable(main_executable_posix src/posix_main.c)
target_link_libraries(main_executable_posix PRIVATE 
    m
    pthread 
    Threads::Threads
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_options(main_executable_posix PRIVATE ${COMMON_COMPILE_OPTIONS})