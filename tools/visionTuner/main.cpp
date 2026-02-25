#include "analyser.hpp"
#include "mainWindow.hpp"
#include "webcamAnalysisLoop.hpp"

#include <QApplication>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>

static std::filesystem::path resolveInputPath(const int argc, char** argv) {
	if (argc > 1) {
		return std::filesystem::path(argv[1]);
	}
	return std::filesystem::path(PATH_TEST_IMG) / "angled_hard/angle_1.jpeg";
}

static cv::Mat loadFallbackImage(const int argc, char** argv) {
	const auto inputPath = resolveInputPath(argc, argv);
	cv::Mat image        = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cerr << "Failed to load fallback image: " << inputPath << "\n";
	}
	return image;
}

int main(int argc, char** argv) {
	QApplication application(argc, argv);

	tengen::vision::Analyser analyser;
	tengen::MainWindow window;
	window.resize(1400, 900);
	cv::Mat fallbackImage;

	tengen::WebcamAnalysisLoop webcamLoop(window, analyser, 0, 500);
	window.setPipelineStepChangedCallback([&webcamLoop](const tengen::PipelineStep) { webcamLoop.refreshFromLastFrame(); });

	if (!webcamLoop.start()) {
		std::cerr << "Failed to open webcam (camera index 0). Falling back to static image input.\n";
		fallbackImage = loadFallbackImage(argc, argv);
		window.setPipelineStepChangedCallback(
		        [&window, &analyser, &fallbackImage](const tengen::PipelineStep step) { window.setImage(analyser.analyse(fallbackImage, step)); });
		window.setImage(analyser.analyse(fallbackImage, window.selectedPipelineStep()));
	}

	window.show();

	const int exitCode = application.exec();
	webcamLoop.stop();
	return exitCode;
}
