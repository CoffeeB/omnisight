#pragma once

#include "omnisight/geometry.hpp"
#include <vector>
#include <cstdint>

namespace omnisight {

struct NMSParams {
    float iou_threshold{0.5f};
    float score_threshold{0.05f};
    int max_output_boxes{300};
    bool soft_nms{false};
    float soft_nms_sigma{0.5f};
};

std::vector<int> fast_nms_cpu(
    const std::vector<BoundingBox2D>& boxes,
    const NMSParams& params
);

std::vector<int> fast_rotated_nms_cpu(
    const std::vector<RotatedBox>& boxes,
    const std::vector<float>& scores,
    const NMSParams& params
);

} // namespace omnisight
