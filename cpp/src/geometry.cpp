#include "omnisight/geometry.hpp"
#include <cmath>
#include <algorithm>

namespace omnisight {

std::array<Point2D, 4> RotatedBox::get_corners() const {
    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);
    float hw = width * 0.5f;
    float hh = height * 0.5f;

    // Relative corners before rotation
    float dx1 = -hw * cos_a - -hh * sin_a;
    float dy1 = -hw * sin_a + -hh * cos_a;

    float dx2 = hw * cos_a - -hh * sin_a;
    float dy2 = hw * sin_a + -hh * cos_a;

    float dx3 = hw * cos_a - hh * sin_a;
    float dy3 = hw * sin_a + hh * cos_a;

    float dx4 = -hw * cos_a - hh * sin_a;
    float dy4 = -hw * sin_a + hh * cos_a;

    return {
        Point2D{cx + dx1, cy + dy1},
        Point2D{cx + dx2, cy + dy2},
        Point2D{cx + dx3, cy + dy3},
        Point2D{cx + dx4, cy + dy4}
    };
}

float compute_iou_axis_aligned(const BoundingBox2D& a, const BoundingBox2D& b) {
    float inter_xmin = std::max(a.xmin, b.xmin);
    float inter_ymin = std::max(a.ymin, b.ymin);
    float inter_xmax = std::min(a.xmax, b.xmax);
    float inter_ymax = std::min(a.ymax, b.ymax);

    float inter_w = std::max(0.0f, inter_xmax - inter_xmin);
    float inter_h = std::max(0.0f, inter_ymax - inter_ymin);
    float inter_area = inter_w * inter_h;

    float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 1e-6f) return 0.0f;
    return inter_area / union_area;
}

float compute_polygon_area(const std::vector<Point2D>& poly) {
    if (poly.size() < 3) return 0.0f;
    float area = 0.0f;
    size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    return std::abs(area) * 0.5f;
}

static bool is_inside(const Point2D& p, const Point2D& cp1, const Point2D& cp2) {
    return (cp2.x - cp1.x) * (p.y - cp1.y) > (cp2.y - cp1.y) * (p.x - cp1.x);
}

static Point2D intersection_point(const Point2D& cp1, const Point2D& cp2, const Point2D& s, const Point2D& e) {
    Point2D dc{cp1.x - cp2.x, cp1.y - cp2.y};
    Point2D dp{s.x - e.x, s.y - e.y};
    float n1 = cp1.x * cp2.y - cp1.y * cp2.x;
    float n2 = s.x * e.y - s.y * e.x;
    float n3 = 1.0f / (dc.x * dp.y - dc.y * dp.x + 1e-8f);
    return Point2D{(n1 * dp.x - n2 * dc.x) * n3, (n1 * dp.y - n2 * dc.y) * n3};
}

std::vector<Point2D> clip_polygon_sutherland_hodgman(
    const std::vector<Point2D>& subject,
    const std::vector<Point2D>& clip
) {
    std::vector<Point2D> output = subject;
    if (clip.empty() || subject.empty()) return {};

    for (size_t i = 0; i < clip.size(); ++i) {
        Point2D cp1 = clip[i];
        Point2D cp2 = clip[(i + 1) % clip.size()];

        std::vector<Point2D> input = output;
        output.clear();
        if (input.empty()) break;

        Point2D s = input.back();
        for (const auto& e : input) {
            if (is_inside(e, cp1, cp2)) {
                if (is_inside(s, cp1, cp2)) {
                    output.push_back(e);
                } else {
                    output.push_back(intersection_point(cp1, cp2, s, e));
                    output.push_back(e);
                }
            } else if (is_inside(s, cp1, cp2)) {
                output.push_back(intersection_point(cp1, cp2, s, e));
            }
            s = e;
        }
    }
    return output;
}

float compute_rotated_iou(const RotatedBox& box_a, const RotatedBox& box_b) {
    auto corners_a = box_a.get_corners();
    auto corners_b = box_b.get_corners();

    std::vector<Point2D> poly_a(corners_a.begin(), corners_a.end());
    std::vector<Point2D> poly_b(corners_b.begin(), corners_b.end());

    std::vector<Point2D> inter_poly = clip_polygon_sutherland_hodgman(poly_a, poly_b);
    float inter_area = compute_polygon_area(inter_poly);

    float union_area = box_a.area() + box_b.area() - inter_area;
    if (union_area <= 1e-6f) return 0.0f;
    return inter_area / union_area;
}

} // namespace omnisight
