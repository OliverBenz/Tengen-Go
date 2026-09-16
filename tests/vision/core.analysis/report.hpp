#pragma once

#include "measure.hpp"

namespace tengen::vision::analysis {

//! Each image against its labelled ground truth.
void printLabelledReport(const ImageSet& set, const std::vector<Measurement>& measurements);

//! Each frame against the set's own median quad. A fixed camera should not move, so drift means a failure.
void printSeriesReport(const ImageSet& set, const std::vector<Measurement>& measurements);

} // namespace tengen::vision::analysis
