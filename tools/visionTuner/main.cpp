#include "analyser.hpp"
#include "mainWindow.hpp"

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

int main(int argc, char** argv) {
	QApplication application(argc, argv);

	const auto inputPath     = resolveInputPath(argc, argv);
	const cv::Mat inputImage = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
	if (inputImage.empty()) {
		std::cerr << "Failed to load image: " << inputPath << "\n";
	}

	const tengen::vision::Analyser analyser(inputImage);

	tengen::MainWindow window;
	window.resize(1400, 900);
	window.setPipelineStepChangedCallback([&window, &analyser](const tengen::PipelineStep step) { window.setImage(analyser.analyse(step)); });
	window.setImage(analyser.analyse(window.selectedPipelineStep()));
	window.show();

	return application.exec();
}
