include_guard(GLOBAL)

set(targetName gameRuntime)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/include/tengen/sessionManager.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/tengen/IAppSignal.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/tengen/IGameSession.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/tengen/IChatSession.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/tengen/INetworkSession.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/tengen/gameServer.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/botSession.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/networkSession.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/openSession.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/eventHub.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/sessionGameInfo.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/sessionChatInfo.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/sessionNetworkInfo.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/logging.hpp"
)

set(sources
	"${CMAKE_CURRENT_LIST_DIR}/src/botSession.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/networkSession.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/openSession.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/sessionManager.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/eventHub.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/sessionGameInfo.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/sessionChatInfo.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/sessionNetworkInfo.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gameServer.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/logging.cpp"
)

add_library(${targetName} STATIC ${headers} ${sources})
add_library(tengen::game::runtime ALIAS ${targetName})

target_include_directories(${targetName}
	PUBLIC
		"${CMAKE_CURRENT_LIST_DIR}/include"
)

# TODO: GameNet should be private.
target_link_libraries(${targetName}
	PRIVATE
		Logger::Logger
	PUBLIC
		tengen::game::core
		tengen::game::engine
		tengen::net::gameNet
)

set_target_properties(${targetName}
	PROPERTIES
		FOLDER "${IDE_FOLDER_SOURCE}/game"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
