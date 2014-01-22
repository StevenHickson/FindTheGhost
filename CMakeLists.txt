cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

IF(WIN32 OR win64)
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll)
ELSE()
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()

find_package( OpenCV REQUIRED )

project(FindTheGhost)

add_subdirectory(src)

include_directories(C:/opencv/build)
include_directories(${FindTheGhost_SOURCE_DIR}/include)

file(GLOB_RECURSE FindTheGhost_HEADERS include/*.h)
file(GLOB_RECURSE FindTheGhost_SOURCES src/*.cpp)
set (FindTheGhost_INCLUDE_DIRS "")
foreach (_headerFile ${FindTheGhost_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND FindTheGhost_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES FindTheGhost_INCLUDE_DIRS)

include_directories(${FindTheGhost_INCLUDE_DIRS})

add_executable (FindTheGhost ${FindTheGhost_SOURCES} ${FindTheGhost_HEADERS})
target_link_libraries ( FindTheGhost ${OpenCV_LIBS})