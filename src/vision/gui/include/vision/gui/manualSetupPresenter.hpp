#pragma once

#include "vision/manualSetup.hpp"

#include <QImage>
#include <QObject>
#include <QPointF>

namespace tengen::vision::gui {

class ManualSetupView;

//! Drives the manual setup: pick the corners on the live camera image, then watch the pipeline work through them.
class ManualSetupPresenter final : public QObject {
	Q_OBJECT

public:
	ManualSetupPresenter(ManualSetupView& view, int cameraIndex);

private:
	ManualSetup::Callbacks callbacks(); //!< Report to this presenter, on the GUI thread.

	void onCornerPicked(QPointF corner);
	void onApplyRequested();
	void onRedoRequested();

	// GUI thread only.
	void showPicking(); //!< Start over: no image until the next camera frame, no corners.
	void showFrame(const QImage& frame);
	void showSnapshot(const QImage& warped, const QImage& rectified, unsigned boardSize);

private:
	ManualSetupView& m_view;
	ManualSetup m_setup; //!< Last: its camera stops before the members it reports to go.
};

} // namespace tengen::vision::gui
