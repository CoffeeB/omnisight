#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <concepts>

namespace omnisight {

struct Point2D {
    float x{0.0f};
    float y{0.0f};

    constexpr Point2D() = default;
    constexpr Point2D(float _x, float _y) : x(_x), y(_y) {}

    constexpr Point2D operator+(const Point2D& other) const { return {x + other.x, y + other.y}; }
    constexpr Point2D operator-(const Point2D& other) const { return {x - other.x, y - other.y}; }
    constexpr Point2D operator*(float s) const { return {x * s, y * s}; }
};

struct RotatedBox {
    float cx{0.0f};
    float cy{0.0f};
    float width{0.0f};
    float height{0.0f};
    float angle_rad{0.0f}; // Rotation angle in radians

    constexpr RotatedBox() = default;
    constexpr RotatedBox(float _cx, float _cy, float _w, float _h, float _angle)
        : cx(_cx), cy(_cy), width(_w), height(_h), angle_rad(_angle) {}

    [[nodiscard]] std::array<Point2D, 4> get_corners() const;
    [[nodiscard]] float area() const { return width * height; }
};

struct BoundingBox2D {
    float xmin{0.0f};
    float ymin{0.0f};
    float xmax{0.0f};
    float ymax{0.0f};
    float score{0.0f};
    int class_id{0};

    constexpr BoundingBox2D() = default;
    constexpr BoundingBox2D(float _xmin, float _ymin, float _xmax, float _ymax, float _s = 1.0f, int _c = 0)
        : xmin(_xmin), ymin(_ymin), xmax(_xmax), ymax(_ymax), score(_s), class_id(_c) {}

    [[nodiscard]] constexpr float area() const {
        float w = std::max(0.0f, xmax - xmin);
        float h = std::max(0.0f, ymax - ymin);
        return w * h;
    }
};

// C++20 Concepts
template<typename T>
concept GeometryFloat = std::floating_point<T>;

// Geometric Functions
float compute_iou_axis_aligned(const BoundingBox2D& box_a, const BoundingBox2D& box_b);
float compute_polygon_area(const std::vector<Point2D>& poly);
std::vector<Point2D> clip_polygon_sutherland_hodgman(const std::vector<Point2D>& subject_polygon, const std::vector<Point2D>& clip_polygon);
float compute_rotated_iou(const RotatedBox& box_a, const RotatedBox& box_b);

} // namespace omnisight
