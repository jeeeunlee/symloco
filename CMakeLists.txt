# we need cmake >= 3.11 to support fetch_content that downloads dependencies
cmake_minimum_required(VERSION 3.15)

# set name of the project
project(symlocorl) # symmetrical locomotion reinforcement learning
set(CMAKE_CXX_STANDARD 17)

# this option is necessary since we need to build a shared library for python
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# release as a default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# data folder path
set(SYMLOCO_DATA_FOLDER
        "\"${CMAKE_CURRENT_LIST_DIR}/data\""
        CACHE STRING ""
        )

# log folder path
set(SYMLOCO_LOG_FOLDER
        "\"${CMAKE_CURRENT_LIST_DIR}/log\""
        CACHE STRING ""
        )

# -----------------------------------------------------------------------------
# options
# -----------------------------------------------------------------------------

option(SYMLOCO_BUILD_APPS "Build symloco apps." ON)

# -----------------------------------------------------------------------------
# macro and functions
# -----------------------------------------------------------------------------

# create static library named LIB_NAME
function(
        create_library #
        LIB_NAME #
        SOURCE #
        DEPENDENCY #
        INCLUDE_DIRS #
        LINK_LIBS #
        COMPILE_DEFINITIONS #
)
    add_library(${LIB_NAME} STATIC ${SOURCE})
    add_library(lib::${LIB_NAME} ALIAS ${LIB_NAME})
    add_dependencies(${LIB_NAME} ${DEPENDENCY})
    target_include_directories(${LIB_NAME} ${INCLUDE_DIRS})
    target_link_libraries(${LIB_NAME} ${LINK_LIBS})

    if (COMPILE_DEFINITIONS)
        # scope is PUBLIC so these definitions are propagated
        target_compile_definitions(${LIB_NAME} ${COMPILE_DEFINITIONS})
    endif ()

    # For solution explorer in visual studios
    set_property(TARGET ${LIB_NAME} PROPERTY FOLDER "Libs")
endfunction()

# create executable named APP_NAME
function(
        create_app #
        APP_NAME #
        SOURCE #
        DEPENDENCY #
        INCLUDE_DIRS #
        LINK_LIBS #
        COMPILE_DEFINITIONS #
)

    add_executable(${APP_NAME} ${SOURCE})
    add_dependencies(${APP_NAME} ${DEPENDENCY})
    target_include_directories(${APP_NAME} ${INCLUDE_DIRS})
    target_link_libraries(${APP_NAME} ${LINK_LIBS})

    if (COMPILE_DEFINITIONS)
        target_compile_definitions(${APP_NAME} ${COMPILE_DEFINITIONS})
    endif ()

    # For solution explorer in visual studios
    set_property(TARGET ${APP_NAME} PROPERTY FOLDER "Apps")
endfunction()

# -----------------------------------------------------------------------------
# code
# -----------------------------------------------------------------------------

# our code
add_subdirectory(src)
include_directories(${PROJECT_SOURCE_DIR}/src)

