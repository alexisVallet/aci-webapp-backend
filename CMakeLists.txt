cmake_minimum_required(VERSION 2.6)
project(ACI)

# with -fPIC
IF(UNIX AND NOT WIN32)
  FIND_PROGRAM(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin )
  IF(CMAKE_UNAME)
    EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR)
    SET(CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR} CACHE INTERNAL
"processor type (i386 and x86_64)")
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      ADD_DEFINITIONS(-fPIC)
    ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  ENDIF(CMAKE_UNAME)
ENDIF(UNIX AND NOT WIN32)

# main library
find_package(OpenCV REQUIRED)
include_directories("${PROJECT_SOURCE_DIR}")
add_library(ACI SHARED aci.cpp)
add_library(IdentityClassifier IdentityClassifier.cpp)
target_link_libraries(ACI IdentityClassifier ${OpenCV_LIBS})

install(FILES aci.h DESTINATION include)
install(TARGETS ACI DESTINATION lib)

#tests
enable_testing()
add_executable(GrabCutExe test/GrabcutTest.cpp)
target_link_libraries(GrabCutExe ${OpenCV_LIBS})
add_test(GrabCutTest GrabCutExe "test/testimage.jpg")
