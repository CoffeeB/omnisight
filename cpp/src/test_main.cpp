#include "omnisight/geometry.hpp"
#include "omnisight/fast_nms.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[OMNISIGHT C++20 Core] Running Geometry & NMS verification..." << std::endl;

    // Test 1: Axis-aligned IoU
    omnisight::BoundingBox2D b1(0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 1);
    omnisight::BoundingBox2D b2(5.0f, 0.0f, 15.0f, 10.0f, 0.8f, 1);
    float iou_aa = omnisight::compute_iou_axis_aligned(b1, b2);
    std::cout << "Axis-aligned IoU (expected ~0.333): " << iou_aa << std::endl;
    assert(std::abs(iou_aa - (50.0f / 150.0f)) < 1e-4f);

    // Test 2: Rotated Box IoU (two identical squares with 0 rotation)
    omnisight::RotatedBox r1(5.0f, 5.0f, 10.0f, 10.0f, 0.0f);
    omnisight::RotatedBox r2(5.0f, 5.0f, 10.0f, 10.0f, 0.0f);
    float iou_rot = omnisight::compute_rotated_iou(r1, r2);
    std::cout << "Rotated IoU identical (expected 1.0): " << iou_rot << std::endl;
    assert(std::abs(iou_rot - 1.0f) < 1e-3f);

    // Test 3: Fast NMS
    std::vector<omnisight::BoundingBox2D> boxes = {
        omnisight::BoundingBox2D(0.0f, 0.0f, 10.0f, 10.0f, 0.95f, 1),
        omnisight::BoundingBox2D(1.0f, 1.0f, 10.0f, 10.0f, 0.85f, 1), // heavy overlap
        omnisight::BoundingBox2D(50.0f, 50.0f, 60.0f, 60.0f, 0.90f, 2) // disjoint
    };
    omnisight::NMSParams params;
    params.iou_threshold = 0.5f;
    params.score_threshold = 0.1f;
    auto keep = omnisight::fast_nms_cpu(boxes, params);

    std::cout << "NMS kept indices count (expected 2): " << keep.size() << std::endl;
    assert(keep.size() == 2);
    assert(keep[0] == 0);
    assert(keep[1] == 2);

    std::cout << "[OMNISIGHT C++20 Core] All geometric and NMS assertions passed successfully!" << std::endl;
    return 0;
}
