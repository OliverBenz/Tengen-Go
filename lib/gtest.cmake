cmake_minimum_required(VERSION 3.14)

include(FetchContent)

# GTest
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # For Windows: Prevent overriding the parent project's compiler/linker settings
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)          # Keep GTest out of our install

FetchContent_MakeAvailable(googletest)
set_target_properties(gtest      PROPERTIES FOLDER "${IDE_FOLDER_EXTERNAL}")
set_target_properties(gtest_main PROPERTIES FOLDER "${IDE_FOLDER_EXTERNAL}")
set_target_properties(gmock      PROPERTIES FOLDER "${IDE_FOLDER_EXTERNAL}")
set_target_properties(gmock_main PROPERTIES FOLDER "${IDE_FOLDER_EXTERNAL}")
