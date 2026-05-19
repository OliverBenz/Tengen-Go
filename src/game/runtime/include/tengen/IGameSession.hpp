#pragma once

#include "tengen/IAppSignalListener.hpp"
#include "model/board.hpp"
#include "model/gameStatus.hpp"

namespace tengen::app {

class IGameSession {
public:
    virtual ~IGameSession() = default;

    virtual void subscribe(app::IAppSignalListener* listener, uint64_t mask) = 0;
    virtual void unsubscribe(app::IAppSignalListener* listener) = 0;

    virtual GameStatus status() const = 0;
    virtual Board board() const = 0;
    virtual Player currentPlayer() const = 0;

    virtual void tryPlace(unsigned x, unsigned y) = 0;
    virtual void tryPass() = 0;
    virtual void tryResign() = 0;
    virtual void shutdown() = 0;
};


}
