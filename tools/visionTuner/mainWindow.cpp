#include "mainWindow.hpp"

#include <QPainter>

#include <opencv2/imgproc.hpp>

namespace tengen {

CvMatrixView::CvMatrixView(QWidget* parent) : QWidget(parent) {
}

void CvMatrixView::setMat(const cv::Mat& mat) {
	m_image = matToQImage(mat);
	update();
}

void CvMatrixView::paintEvent(QPaintEvent* event) {
	QWidget::paintEvent(event);

	QPainter painter(this);
	painter.fillRect(rect(), Qt::black);

	if (m_image.isNull()) {
		return;
	}

	const QImage scaled = m_image.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	const QPoint topLeft((width() - scaled.width()) / 2, (height() - scaled.height()) / 2);
	painter.drawImage(topLeft, scaled);
}

QImage CvMatrixView::matToQImage(const cv::Mat& mat) {
	if (mat.empty()) {
		return {};
	}

	switch (mat.type()) {
	case CV_8UC1: {
		const QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
		return image.copy();
	}
	case CV_8UC3: {
		cv::Mat rgb;
		cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
		const QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
		return image.copy();
	}
	case CV_8UC4: {
		cv::Mat rgba;
		cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
		const QImage image(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step), QImage::Format_RGBA8888);
		return image.copy();
	}
	default:
		break;
	}

	cv::Mat normalized8;
	if (mat.depth() == CV_8U) {
		normalized8 = mat;
	} else {
		cv::Mat normalizedFloat;
		cv::normalize(mat, normalizedFloat, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);
		normalizedFloat.convertTo(normalized8, CV_8U);
	}

	switch (normalized8.channels()) {
	case 1: {
		const QImage image(normalized8.data, normalized8.cols, normalized8.rows, static_cast<int>(normalized8.step), QImage::Format_Grayscale8);
		return image.copy();
	}
	case 3: {
		cv::Mat rgb;
		cv::cvtColor(normalized8, rgb, cv::COLOR_BGR2RGB);
		const QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
		return image.copy();
	}
	case 4: {
		cv::Mat rgba;
		cv::cvtColor(normalized8, rgba, cv::COLOR_BGRA2RGBA);
		const QImage image(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step), QImage::Format_RGBA8888);
		return image.copy();
	}
	default:
		return {};
	}
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
	setWindowTitle("Vision Tuner");
	buildLayout();
}

MainWindow::~MainWindow() = default;

void MainWindow::setImage(const cv::Mat& image) {
	if (m_matrixView != nullptr) {
		m_matrixView->setMat(image);
	}
}

void MainWindow::buildLayout() {
	m_matrixView = new CvMatrixView(this);
	setCentralWidget(m_matrixView);
}

} // namespace tengen
