#include "omnisight/fast_nms.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace omnisight {

std::vector<int> fast_nms_cpu(
    const std::vector<BoundingBox2D>& boxes,
    const NMSParams& params
) {
    if (boxes.empty()) return {};

    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Filter by score threshold and sort descending by score
    std::vector<int> valid_indices;
    valid_indices.reserve(boxes.size());
    for (int idx : indices) {
        if (boxes[idx].score >= params.score_threshold) {
            valid_indices.push_back(idx);
        }
    }

    std::sort(valid_indices.begin(), valid_indices.end(), [&](int a, int b) {
        return boxes[a].score > boxes[b].score;
    });

    std::vector<int> keep;
    keep.reserve(std::min((size_t)params.max_output_boxes, valid_indices.size()));

    std::vector<bool> suppressed(valid_indices.size(), false);

    for (size_t i = 0; i < valid_indices.size(); ++i) {
        if (suppressed[i]) continue;
        int current_idx = valid_indices[i];
        keep.push_back(current_idx);
        if (static_cast<int>(keep.size()) >= params.max_output_boxes) break;

        for (size_t j = i + 1; j < valid_indices.size(); ++j) {
            if (suppressed[j]) continue;
            int next_idx = valid_indices[j];
            float iou = compute_iou_axis_aligned(boxes[current_idx], boxes[next_idx]);
            if (iou >= params.iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

std::vector<int> fast_rotated_nms_cpu(
    const std::vector<RotatedBox>& boxes,
    const std::vector<float>& scores,
    const NMSParams& params
) {
    if (boxes.empty()) return {};

    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<int> valid_indices;
    valid_indices.reserve(boxes.size());
    for (int idx : indices) {
        if (scores[idx] >= params.score_threshold) {
            valid_indices.push_back(idx);
        }
    }

    std::sort(valid_indices.begin(), valid_indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<int> keep;
    keep.reserve(std::min((size_t)params.max_output_boxes, valid_indices.size()));
    std::vector<bool> suppressed(valid_indices.size(), false);

    for (size_t i = 0; i < valid_indices.size(); ++i) {
        if (suppressed[i]) continue;
        int current_idx = valid_indices[i];
        keep.push_back(current_idx);
        if (static_cast<int>(keep.size()) >= params.max_output_boxes) break;

        for (size_t j = i + 1; j < valid_indices.size(); ++j) {
            if (suppressed[j]) continue;
            int next_idx = valid_indices[j];
            float iou = compute_rotated_iou(boxes[current_idx], boxes[next_idx]);
            if (iou >= params.iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

} // namespace omnisight
