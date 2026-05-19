#pragma once

#include "tengen/IAppSignal.hpp"
#include "model/board.hpp"
#include "model/gameStatus.hpp"

namespace tengen::app {

class IGameSession : public IAppSignalSource {
public:
    virtual ~IGameSession() = default;

    virtual GameStatus status() const = 0;
    virtual Board board() const = 0;
    virtual Player currentPlayer() const = 0;

    virtual void tryPlace(unsigned x, unsigned y) = 0;
    virtual void tryPass() = 0;
    virtual void tryResign() = 0;
    virtual void shutdown() = 0;
};


}
