include_guard(GLOBAL)

set(targetName netNetwork)

# Get files to build
set(headers
    "${CMAKE_CURRENT_LIST_DIR}/include/network/client.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/network/server.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/network/nwEvents.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/network/types.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/clientImpl.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/sessionManager.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/serverEvents.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/serverImpl.hpp"
)

set(sources
    "${CMAKE_CURRENT_LIST_DIR}/src/client.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/clientImpl.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/server.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/serverImpl.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/nwEvents.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/sessionManager.cpp"
)


add_library(${targetName} STATIC ${headers} ${sources})
add_library(tengen::net::gameNet ALIAS ${targetName})

target_include_directories(${targetName}
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
)

target_link_libraries(${targetName}
    PUBLIC
        tengen::game::model
    PRIVATE
        tengen::net::core
        nlohmann_json::nlohmann_json
)

set_target_properties(${targetName}
     PROPERTIES
         FOLDER "${IDE_FOLDER_SOURCE}/net"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
