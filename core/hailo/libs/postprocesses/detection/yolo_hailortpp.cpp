#include "hailo_nms_decode.hpp"
#include "yolo_hailortpp.hpp"
#include "common/labels/coco_eighty.hpp"

static const std::string DEFAULT_YOLOV5M_OUTPUT_LAYER = "yolov5_nms_postprocess";

static std::map<uint8_t, std::string> yolo_vehicles_labels = {
    {0, "unlabeled"},
    {1, "car"}};

static std::map<uint8_t, std::string> yolo_mvigs_labels = {
    {0, "unlabeled"},
    {1, "persons"},
    {2, "cars"},
    {3, "motorcycles"},
    {4, "buses"}};


void yolov5(HailoROIPtr roi)
{
    //auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), yolo_mvigs_labels);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolox(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor("yolox_nms_postprocess"), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5m_vehicles(HailoROIPtr roi)
{
    // auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), yolo_vehicles_labels, 0.4, DEFAULT_MAX_BOXES, true);
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), yolo_vehicles_labels);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    hailo_common::add_detections(roi, detections);
}

void yolov5_no_persons(HailoROIPtr roi)
{
    auto post = HailoNMSDecode(roi->get_tensor(DEFAULT_YOLOV5M_OUTPUT_LAYER), common::coco_eighty);
    auto detections = post.decode<float32_t, common::hailo_bbox_float32_t>();
    for (auto it = detections.begin(); it != detections.end();)
    {
        if (it->get_label() == "person")
        {
            it = detections.erase(it);
        }
        else
        {
            ++it;
        }
    }
    hailo_common::add_detections(roi, detections);
}

void filter(HailoROIPtr roi)
{
    yolov5(roi);
}
