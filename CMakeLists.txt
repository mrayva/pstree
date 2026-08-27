cmake_minimum_required(VERSION 3.20)

project(pstree
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "PS-Tree / PSTDynamic boolean expression matching, from Ji et al. 2025"
)

option(BUILD_EXAMPLES "Build usage examples" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cmake/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cmake/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cmake/bin)

add_library(pstree INTERFACE)
add_library(pstree::pstree ALIAS pstree)
target_include_directories(pstree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(pstree INTERFACE cxx_std_20)

enable_testing()

function(add_pstree_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE pstree)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

set(PSTREE_UNIT_TESTS
    test_order_key
    test_ps_tree
)

foreach(test_name IN LISTS PSTREE_UNIT_TESTS)
    add_pstree_test(${test_name})
endforeach()

message(STATUS "")
message(STATUS "pstree configuration:")
message(STATUS "  BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
message(STATUS "  C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "")
