#include "engine/gtpEngine.hpp"

#include "gtpEngineImpl.hpp"

#include <memory>
#include <utility>

namespace tengen::engine {

GtpEngine::GtpEngine()
    : m_pimpl(std::make_unique<Implementation>()) {
}

GtpEngine::~GtpEngine() {
	stop();
}

bool GtpEngine::registerListener(IEngineListener* listener) {
	return m_pimpl->registerListener(listener);
}

void GtpEngine::stop() {
	m_pimpl->stop();
}

bool GtpEngine::isRunning() const {
	return m_pimpl->isRunning();
}

bool GtpEngine::place(const tengen::Coord pos) {
	return m_pimpl->place(pos);
}

bool GtpEngine::pass() {
	return m_pimpl->pass();
}

void GtpEngine::genmove() {
	m_pimpl->genmove();
}

void GtpEngine::launch(Launch command, const GameConfig& gameConfig, const tengen::Player botColour) {
	m_pimpl->start(std::move(command), gameConfig, botColour);
}

} // namespace tengen::engine
