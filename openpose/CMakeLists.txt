cmake_minimum_required(VERSION 3.0)

project(testopenpose)
set( CMAKE_CXX_FLAGS " -std=c++11" )

file(GLOB openpose_source
     *.cpp
     *.cu
     )
cuda_add_executable(testopenpose ${openpose_source})
set_property(TARGET testopenpose PROPERTY CXX_STANDARD 17)
# target_link_libraries(testopenpose tinytrt)
target_link_libraries(testopenpose nvinfer)
target_include_directories(testopenpose PRIVATE ${TENSORRT_INCLUDE_DIR})
target_include_directories(testopenpose PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_directories(testopenpose PRIVATE ${OpenCV_DIR})
# TODO: automatically infer version number from OpenCV config
target_link_libraries(testopenpose opencv_world451)
target_link_directories(testopenpose PRIVATE ${tinytrt_BINARY_DIR}/${CMAKE_BUILD_TYPE})
target_link_directories(testopenpose PRIVATE ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
target_link_libraries(testopenpose tinytrt)
target_include_directories(testopenpose PRIVATE ${OpenPose_SOURCE_DIR}/include)
target_include_directories(testopenpose PRIVATE ${OpenPose_SOURCE_DIR}/3rdparty/windows/caffe3rdparty/include)
target_link_directories(testopenpose PRIVATE ${OpenPose_SOURCE_DIR}/3rdparty/windows/caffe3rdparty/lib)
target_link_libraries(testopenpose gflags)
target_link_directories(testopenpose PRIVATE ${OpenPose_BINARY_DIR}/src/openpose/${CMAKE_BUILD_TYPE})
target_link_libraries(testopenpose openpose)

set(Boost_DIR "\\\\BVIA-BUILD\\boost\\1_75_0" CACHE STRING "Path to a file." FORCE)
set(Boost_INCLUDE_DIR "\\\\BVIA-BUILD\\boost\\1_75_0" CACHE STRING "Path to a file." FORCE)
set(Boost_SERIALIZATION_LIBRARY_DEBUG "\\\\BVIA-BUILD\\boost\\1_75_0\\lib64-msvc-14.2\\boost_serialization-vc142-mt-gd-x64-1_75.lib" CACHE STRING "Boost serialization library (debug)" FORCE)
set(Boost_SERIALIZATION_LIBRARY_RELEASE "\\\\BVIA-BUILD\\boost\\1_75_0\\lib64-msvc-14.2\\boost_serialization-vc142-mt-x64-1_75.lib" CACHE STRING "Boost serialization library (release)" FORCE)
find_package(Boost 1.75 REQUIRED serialization)
if (Boost_FOUND)
	target_include_directories(testopenpose PRIVATE ${Boost_INCLUDE_DIRS})
	target_link_directories(testopenpose PRIVATE ${Boost_LIBRARY_DIR_RELEASE}) # not necessary, Boost_SERIALIZATION_LIBRARY_RELEASE contains complete path
	target_link_libraries(testopenpose ${Boost_SERIALIZATION_LIBRARY_RELEASE})
endif()
target_compile_definitions(testopenpose PUBLIC BOOST_ALL_NO_LIB=1)
if (MSVC)
	set(CMAKE_CXX_FLAGS "/EHsc")
endif()

# target_link_libraries(testopenpose nvinfer_plugin)
# target_link_libraries(testopenpose nvparsers)
# target_link_libraries(testopenpose nvonnxparser)
target_link_libraries(testopenpose nvcaffeparser)
# target_link_libraries(testopenpose ${CUDART})
# target_link_libraries(testopenpose ${OpenCV_LIBRARIES})
