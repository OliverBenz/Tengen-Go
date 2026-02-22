#pragma once

#include <QImage>
#include <QMainWindow>
#include <QWidget>

#include <opencv2/core/mat.hpp>

namespace tengen {

class CvMatrixView : public QWidget {
public:
	explicit CvMatrixView(QWidget* parent = nullptr);
	void setMat(const cv::Mat& mat);

protected:
	void paintEvent(QPaintEvent* event) override;

private:
	static QImage matToQImage(const cv::Mat& mat);

	QImage m_image{};
};

class MainWindow : public QMainWindow {
public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow() override;

	void setImage(const cv::Mat& image);

private:
	void buildLayout();

private:
	CvMatrixView* m_matrixView{nullptr};
};

} // namespace tengen
