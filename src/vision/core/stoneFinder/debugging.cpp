#include "debugging.hpp"

#include "stoneFinderInternal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>

#if defined(VISION_DEBUG_LOGGING) && defined(VISION_LOG_STONEFINDER)
#include <format>
#include <iostream>
#define DEBUG_LOG(x) std::cout << x

namespace {

const char* rejectionReasonLabel(tengen::vision::core::RejectionReason reason) {
	switch (reason) {
	case tengen::vision::core::RejectionReason::None:
		return "None";
	case tengen::vision::core::RejectionReason::WeakZ:
		return "WeakZ";
	case tengen::vision::core::RejectionReason::LowConfidence:
		return "LowConfidence";
	case tengen::vision::core::RejectionReason::WeakSupport:
		return "WeakSupport";
	case tengen::vision::core::RejectionReason::WeakNeighborContrast:
		return "WeakNeighborContrast";
	case tengen::vision::core::RejectionReason::EdgeArtifact:
		return "EdgeArtifact";
	case tengen::vision::core::RejectionReason::MarginTooSmall:
		return "MarginTooSmall";
	case tengen::vision::core::RejectionReason::Other:
		return "Other";
	}
	return "Other";
}

} // namespace
#else
#define DEBUG_LOG(x) ((void)0)
#endif

namespace tengen::vision::core {
namespace Debugging {

bool isRuntimeDebugEnabled() {
	const char* debugEnv = std::getenv("GO_STONE_DEBUG");
	if (debugEnv == nullptr) {
		return false;
	}
	const std::string_view debugFlag(debugEnv);
	return debugFlag == "1" || debugFlag == "2";
}

cv::Mat drawOverlay(const cv::Mat& image, const std::vector<cv::Point2f>& intersections, const std::vector<StoneState>& states, int radius) {
	cv::Mat overlay = image.clone();
	for (std::size_t index = 0; index < intersections.size() && index < states.size(); ++index) {
		if (states[index] == StoneState::Black) {
			cv::circle(overlay, intersections[index], radius, cv::Scalar(0, 0, 0), 2);
		} else if (states[index] == StoneState::White) {
			cv::circle(overlay, intersections[index], radius, cv::Scalar(255, 0, 0), 2);
		}
	}
	return overlay;
}

cv::Mat renderStatsTile(const Model& model, const DebugStats& stats) {
	cv::Mat tile(220, 450, CV_8UC3, cv::Scalar(255, 255, 255));
	int y              = 24;
	const auto putLine = [&](const std::string& line) {
		cv::putText(tile, line, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		y += 22;
	};

	putLine("Stone Detection v2");
	putLine(std::format("medianEmpty: {:.2f}", model.medianEmpty));
	putLine(std::format("sigmaEmpty: {:.2f}", model.sigmaEmpty));
	putLine(std::format("chromaT: {:.1f}", model.tChromaSq));
	putLine("black: " + std::to_string(stats.blackCount));
	putLine("white: " + std::to_string(stats.whiteCount));
	putLine("empty: " + std::to_string(stats.emptyCount));
	putLine("refine tried: " + std::to_string(stats.refinedTried));
	putLine("refine accepted: " + std::to_string(stats.refinedAccepted));
	return tile;
}

void emitRuntimeDebug(const BoardGeometry& geometry, const std::vector<Features>& features, const Model& model, const std::vector<StoneState>& states,
                      const std::vector<float>& confidence, const std::vector<Eval>& evaluations, const std::vector<float>& neighborMedianMap,
                      const DebugStats& stats, const std::vector<RejectionReason>* rejectionReasons) {
#if defined(VISION_DEBUG_LOGGING) && defined(VISION_LOG_STONEFINDER)
	const int boardSize        = static_cast<int>(geometry.boardSize);
	const bool hasEvaluations  = evaluations.size() == states.size();
	const bool hasNeighborMeds = neighborMedianMap.size() == states.size();
	const auto zForIndex       = [&](std::size_t index, const Features& feature) {
        return hasEvaluations ? evaluations[index].z : (feature.deltaL - model.medianEmpty) / model.sigmaEmpty;
	};
	const auto rawStateForIndex = [&](std::size_t index) {
		if (!hasEvaluations) {
			return StoneState::Empty;
		}
		return evaluations[index].state;
	};
	const auto neighborForIndex = [&](std::size_t index) {
		if (hasNeighborMeds) {
			return neighborMedianMap[index];
		}
		const int gridX = static_cast<int>(index) / boardSize;
		const int gridY = static_cast<int>(index) - gridX * boardSize;
		return computeNeighborMedianDelta(features, gridX, gridY, boardSize, model.medianEmpty);
	};

	DEBUG_LOG("[stone-debug] N=" << geometry.boardSize << " black=" << stats.blackCount << " white=" << stats.whiteCount << " empty=" << stats.emptyCount
	                             << " median=" << model.medianEmpty << " sigma=" << model.sigmaEmpty << " chromaT=" << model.tChromaSq
	                             << " refineTried=" << stats.refinedTried << " refineAccepted=" << stats.refinedAccepted);

	for (std::size_t index = 0; index < states.size(); ++index) {
		if (states[index] == StoneState::Empty) {
			continue;
		}
		const std::size_t gridX      = index / geometry.boardSize;
		const std::size_t gridY      = index % geometry.boardSize;
		const Features& feature      = features[index];
		const float z                = zForIndex(index, feature);
		const float neighborMedian   = neighborForIndex(index);
		const float neighborContrast = (states[index] == StoneState::Black) ? (neighborMedian - feature.deltaL) : (feature.deltaL - neighborMedian);
		const cv::Point2f point      = geometry.intersections[index];
		DEBUG_LOG("  idx=" << index << " (" << gridX << "," << gridY << ")"
		                   << " px=(" << point.x << "," << point.y << ")"
		                   << " state=" << (states[index] == StoneState::Black ? "B" : "W") << " conf=" << confidence[index] << " z=" << z
		                   << " d=" << feature.darkFrac << " b=" << feature.brightFrac << " c=" << feature.chromaSq << " nc=" << neighborContrast);
	}

	const bool verboseCandidates = true;
	if (verboseCandidates) {
		struct EmptyRow {
			std::size_t idx{0};
			float z{0.0f};
		};
		std::vector<EmptyRow> emptyRows;
		emptyRows.reserve(features.size());
		for (std::size_t index = 0; index < features.size(); ++index) {
			if (!features[index].valid || states[index] != StoneState::Empty) {
				continue;
			}
			const float z = zForIndex(index, features[index]);
			emptyRows.push_back({index, z});
		}
		std::sort(emptyRows.begin(), emptyRows.end(), [](const EmptyRow& left, const EmptyRow& right) { return left.z > right.z; });
		const std::size_t limit = std::min<std::size_t>(20, emptyRows.size());
		for (std::size_t row = 0; row < limit; ++row) {
			const std::size_t index      = emptyRows[row].idx;
			const std::size_t gridX      = index / geometry.boardSize;
			const std::size_t gridY      = index % geometry.boardSize;
			const StoneState rawState    = rawStateForIndex(index);
			const float rawMargin        = hasEvaluations ? evaluations[index].margin : 0.0f;
			const float rawRequired      = hasEvaluations ? evaluations[index].required : 0.0f;
			const float rawConf          = hasEvaluations ? evaluations[index].confidence : 0.0f;
			const float neighborMedian   = neighborForIndex(index);
			const float neighborContrast = features[index].deltaL - neighborMedian;
			DEBUG_LOG("  empty-cand idx=" << index << " (" << gridX << "," << gridY << ")"
			                              << " z=" << emptyRows[row].z << " d=" << features[index].darkFrac << " b=" << features[index].brightFrac
			                              << " c=" << features[index].chromaSq
			                              << " raw=" << (rawState == StoneState::Black ? "B" : (rawState == StoneState::White ? "W" : "E"))
			                              << " m=" << rawMargin << "/" << rawRequired << " conf=" << rawConf << " nc=" << neighborContrast);
		}

		struct BrightRow {
			std::size_t idx{0};
			float bright{0.0f};
		};
		std::vector<BrightRow> brightRows;
		brightRows.reserve(features.size());
		for (std::size_t index = 0; index < features.size(); ++index) {
			if (!features[index].valid || states[index] != StoneState::Empty) {
				continue;
			}
			brightRows.push_back({index, features[index].brightFrac});
		}
		std::sort(brightRows.begin(), brightRows.end(), [](const BrightRow& left, const BrightRow& right) { return left.bright > right.bright; });
		const std::size_t brightLimit = std::min<std::size_t>(20, brightRows.size());
		for (std::size_t row = 0; row < brightLimit; ++row) {
			const std::size_t index      = brightRows[row].idx;
			const std::size_t gridX      = index / geometry.boardSize;
			const std::size_t gridY      = index % geometry.boardSize;
			const float z                = zForIndex(index, features[index]);
			const StoneState rawState    = rawStateForIndex(index);
			const float rawMargin        = hasEvaluations ? evaluations[index].margin : 0.0f;
			const float rawRequired      = hasEvaluations ? evaluations[index].required : 0.0f;
			const float rawConf          = hasEvaluations ? evaluations[index].confidence : 0.0f;
			const float neighborMedian   = neighborForIndex(index);
			const float neighborContrast = features[index].deltaL - neighborMedian;
			DEBUG_LOG("  bright-cand idx=" << index << " (" << gridX << "," << gridY << ")"
			                               << " b=" << brightRows[row].bright << " z=" << z << " d=" << features[index].darkFrac
			                               << " c=" << features[index].chromaSq
			                               << " raw=" << (rawState == StoneState::Black ? "B" : (rawState == StoneState::White ? "W" : "E"))
			                               << " m=" << rawMargin << "/" << rawRequired << " conf=" << rawConf << " nc=" << neighborContrast);
		}
	}

	if (stats.blackCount + stats.whiteCount == 0) {
		struct CandidateRow {
			std::size_t idx{0};
			float absZ{0.0f};
		};
		std::vector<CandidateRow> rows;
		rows.reserve(features.size());
		for (std::size_t index = 0; index < features.size(); ++index) {
			if (!features[index].valid) {
				continue;
			}
			const float z = zForIndex(index, features[index]);
			rows.push_back({index, std::abs(z)});
		}
		std::sort(rows.begin(), rows.end(), [](const CandidateRow& left, const CandidateRow& right) { return left.absZ > right.absZ; });
		const std::size_t limit = std::min<std::size_t>(10, rows.size());
		for (std::size_t row = 0; row < limit; ++row) {
			const std::size_t index = rows[row].idx;
			const std::size_t gridX = index / geometry.boardSize;
			const std::size_t gridY = index % geometry.boardSize;
			const Features& feature = features[index];
			const float z           = zForIndex(index, feature);
			DEBUG_LOG("  cand idx=" << index << " (" << gridX << "," << gridY << ")"
			                        << " z=" << z << " d=" << feature.darkFrac << " b=" << feature.brightFrac << " c=" << feature.chromaSq);
		}
	}

	if (rejectionReasons != nullptr && rejectionReasons->size() == states.size()) {
		std::array<int, 8> reasonCounts{};
		for (std::size_t index = 0; index < states.size(); ++index) {
			if (states[index] != StoneState::Empty) {
				continue;
			}
			const std::size_t reasonIndex = static_cast<std::size_t>((*rejectionReasons)[index]);
			if (reasonIndex < reasonCounts.size()) {
				++reasonCounts[reasonIndex];
			}
		}
		DEBUG_LOG("[stone-debug] rejections");
		for (std::size_t index = 0; index < reasonCounts.size(); ++index) {
			DEBUG_LOG('\t' << rejectionReasonLabel(static_cast<RejectionReason>(index)) << "=" << reasonCounts[index]);
		}
	}
#else
	(void)geometry;
	(void)features;
	(void)model;
	(void)states;
	(void)confidence;
	(void)evaluations;
	(void)neighborMedianMap;
	(void)stats;
	(void)rejectionReasons;
#endif
}

} // namespace Debugging
} // namespace tengen::vision::core
