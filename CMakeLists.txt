cmake_minimum_required(VERSION 3.5)
project(mppi)

find_package(PythonLibs REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

include_directories(
    guide_mppi/
    model/

    include/EigenRand/
    ${EIGEN3_INCLUDE_DIRS}

    # matplotlibcpp/
    ${PYTHON_INCLUDE_DIRS}

    # 현재 디렉토리의 하위 디렉토리에서 헤더 파일을 찾기 위해
    include/matplotlibcpp/
)

if (quadrotor) 
    set(COMMON_SOURCES
    )
else()
    set(COMMON_SOURCES
        src/wmrobot/guide_mppi.cpp
    )
endif()


foreach(SRC ${COMMON_SOURCES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    target_compile_options(${EXE_NAME} PRIVATE -O3)
    target_link_libraries(${EXE_NAME}     
        ${PYTHON_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
endforeach()