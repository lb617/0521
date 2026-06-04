cmake_minimum_required(VERSION 3.0.0)

project(mouse_face_demo)

# 引入路径配置
include(cmake_config/Paths.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
add_compile_options(-Wno-psabi)

add_executable(mouse_face_demo ${PROJECT_SOURCE_DIR}/demo_mouse_face.cpp)

# 头文件目录
include_directories(${M1_SDK_INC_DIR})
include_directories(${M1_SDK_INC_DIR}/smartsoc)
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SRC_DIR "src")
file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_DIR}/*.cpp")
target_sources(mouse_face_demo PRIVATE ${SRC_FILES})

# 链接 SDK 库
target_link_libraries(${PROJECT_NAME}
    ${M1_SSNE_LIB}
    ${M1_CMABUFFER_LIB}
    ${M1_OSD_LIB}
    ${M1_SSZLOG_LIB}
    ${M1_ZLOG_LIB}
    ${M1_EMB_LIB}
)

install(TARGETS mouse_face_demo DESTINATION bin)
