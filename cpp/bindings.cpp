#ifdef PYBIND11_FOUND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "omnisight/geometry.hpp"
#include "omnisight/fast_nms.hpp"

namespace py = pybind11;

PYBIND11_MODULE(omnisight_accel, m) {
    m.doc() = "OmniSight C++20 High-Performance Geometry and NMS Acceleration Module";

    py::class_<omnisight::BoundingBox2D>(m, "BoundingBox2D")
        .def(py::init<float, float, float, float, float, int>(),
             py::arg("xmin"), py::arg("ymin"), py::arg("xmax"), py::arg("ymax"),
             py::arg("score") = 1.0f, py::arg("class_id") = 0)
        .def_readwrite("xmin", &omnisight::BoundingBox2D::xmin)
        .def_readwrite("ymin", &omnisight::BoundingBox2D::ymin)
        .def_readwrite("xmax", &omnisight::BoundingBox2D::xmax)
        .def_readwrite("ymax", &omnisight::BoundingBox2D::ymax)
        .def_readwrite("score", &omnisight::BoundingBox2D::score)
        .def_readwrite("class_id", &omnisight::BoundingBox2D::class_id)
        .def("area", &omnisight::BoundingBox2D::area);

    py::class_<omnisight::RotatedBox>(m, "RotatedBox")
        .def(py::init<float, float, float, float, float>(),
             py::arg("cx"), py::arg("cy"), py::arg("width"), py::arg("height"), py::arg("angle_rad"))
        .def_readwrite("cx", &omnisight::RotatedBox::cx)
        .def_readwrite("cy", &omnisight::RotatedBox::cy)
        .def_readwrite("width", &omnisight::RotatedBox::width)
        .def_readwrite("height", &omnisight::RotatedBox::height)
        .def_readwrite("angle_rad", &omnisight::RotatedBox::angle_rad)
        .def("area", &omnisight::RotatedBox::area);

    py::class_<omnisight::NMSParams>(m, "NMSParams")
        .def(py::init<>())
        .def_readwrite("iou_threshold", &omnisight::NMSParams::iou_threshold)
        .def_readwrite("score_threshold", &omnisight::NMSParams::score_threshold)
        .def_readwrite("max_output_boxes", &omnisight::NMSParams::max_output_boxes);

    m.def("compute_iou_axis_aligned", &omnisight::compute_iou_axis_aligned, "Compute IoU between two axis-aligned boxes");
    m.def("compute_rotated_iou", &omnisight::compute_rotated_iou, "Compute rotated IoU between two oriented bounding boxes");
    m.def("fast_nms_cpu", &omnisight::fast_nms_cpu, "Fast CPU Non-Maximum Suppression");
    m.def("fast_rotated_nms_cpu", &omnisight::fast_rotated_nms_cpu, "Fast CPU Rotated Non-Maximum Suppression");
}
#endif
