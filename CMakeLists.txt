cmake_minimum_required(VERSION 2.6)
project(ACI)

find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}")
add_library(ACI SHARED aci.cpp)
add_library(IdentityClassifier IdentityClassifier.cpp)
target_link_libraries(ACI IdentityClassifier ${OpenCV_LIBS})

install(FILES aci.h DESTINATION include)
install(TARGETS ACI DESTINATION lib)
