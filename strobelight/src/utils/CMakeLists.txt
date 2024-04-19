cmake_minimum_required(VERSION 3.16)
project(utils)

include_directories(${PROJECT_SOURCE_DIR})

# Tell cmake where to find folly, this is the default path from build.sh script
# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../folly/installed/lib)
# message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")

# Add source files
set(SOURCES
    ProcUtils.cpp
    SymUtils.cpp
)

# Add header files
set(HEADERS
    ProcUtils.h
    SymUtils.h
)

# Add libraries
add_library(utils INTERFACE)

target_sources(utils INTERFACE ${SOURCES} ${HEADERS})

find_package(fmt CONFIG REQUIRED)
target_link_libraries(utils INTERFACE fmt::fmt)

# This needs to be build using build.sh script first
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../blazesym/capi/include)
target_link_libraries(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../../blazesym/target/release/libblazesym_c.a -lpthread -lrt -ldl)
