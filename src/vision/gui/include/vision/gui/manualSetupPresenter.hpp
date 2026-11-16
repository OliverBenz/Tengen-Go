#pragma once

#include "vision/manualSetup.hpp"

#include <QImage>
#include <QObject>
#include <QPointF>

#include <filesystem>

namespace tengen::vision::gui {

class ManualSetupView;

//! Drives the manual setup: pick the corners on the live camera image, then watch the pipeline work through them.
class ManualSetupPresenter final : public QObject {
	Q_OBJECT

public:
	ManualSetupPresenter(ManualSetupView& view, int cameraIndex, const std::filesystem::path& modelPath);

private:
	ManualSetup::Callbacks callbacks(); //!< Report to this presenter, on the GUI thread.

	void onCornerPicked(QPointF corner);
	void onApplyRequested();
	void onRedoRequested();

	// GUI thread only.
	void showPicking();  //!< Start over: no image until the next camera frame, no corners.
	void showProgress(); //!< How many corners are picked, and Apply once all are.
	void showFrame(const QImage& frame);
	void showSnapshot(const QImage& warped, const QImage& rectified, const QString& found);
	void showCameraLost();

private:
	ManualSetupView& m_view;
	ManualSetup m_setup; //!< Last: its camera stops before the members it reports to go.
};

} // namespace tengen::vision::gui
