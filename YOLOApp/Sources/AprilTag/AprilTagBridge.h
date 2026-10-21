#ifndef APRILTAG_BRIDGE_H
#define APRILTAG_BRIDGE_H

// Plain-C surface over the vendored AprilTag library so Swift never touches its internals.

#include <stdint.h>

typedef struct yolo_tag_detector yolo_tag_detector;

typedef struct {
    int id;
    int hamming;
    float decision_margin;
    /// Detected corners in pixels, (x, y) pairs in AprilTag's order (counter-clockwise).
    double corners[8];
    /// Row-major rotation and translation (meters) mapping AprilTag's tag frame
    /// (origin at tag center, x right, y down, z into the tag) into the camera frame.
    double R[9];
    double t[3];
    /// Object-space error reported by the pose solver.
    double pose_error;
} yolo_tag_result;

yolo_tag_detector *yolo_tag_detector_create(int nthreads, float quad_decimate);
void yolo_tag_detector_destroy(yolo_tag_detector *detector);

/// Detects tag36h11 tags in an 8-bit grayscale image and estimates each pose.
/// Writes at most `max_results` results and returns how many were written.
int yolo_tag_detect(yolo_tag_detector *detector,
                    const uint8_t *luma, int width, int height, int stride,
                    double tag_size, double fx, double fy, double cx, double cy,
                    yolo_tag_result *results, int max_results);

#endif
