cmake_minimum_required(VERSION 3.12)
project(yolov9-openvino)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)


# Include CUDA directories

# Add source files
set(SOURCES
    main.cpp
    yolov9_openvino.cpp
)

# Add headers
set(HEADERS
    yolov9_openvino.h
)

# Set your OpenCV path 
set(OpenCV_DIR "C:\\opencv490\\build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set your OpenVINO path 
set(OPENVINO_DIR "C:\\Program Files (x86)\\Intel\\openvino_2023\\runtime")

# Include TensorRT
include_directories(${OPENVINO_DIR}/include)
link_directories(${OPENVINO_DIR}/lib/intel64/release)
set(OPENVINO_LIBS openvino openvino_c)

# Create an executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${OPENVINO_LIBS}
)
