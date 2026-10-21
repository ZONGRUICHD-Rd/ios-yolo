#include "AprilTagBridge.h"

#include <stdlib.h>

#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag36h11.h"
#include "common/matd.h"

struct yolo_tag_detector {
    apriltag_family_t *family;
    apriltag_detector_t *detector;
};

yolo_tag_detector *yolo_tag_detector_create(int nthreads, float quad_decimate) {
    yolo_tag_detector *d = calloc(1, sizeof(yolo_tag_detector));
    d->family = tag36h11_create();
    d->detector = apriltag_detector_create();
    apriltag_detector_add_family_bits(d->detector, d->family, 2);
    d->detector->nthreads = nthreads;
    d->detector->quad_decimate = quad_decimate;
    d->detector->refine_edges = true;
    d->detector->decode_sharpening = 0.25;
    return d;
}

void yolo_tag_detector_destroy(yolo_tag_detector *d) {
    if (!d) return;
    apriltag_detector_destroy(d->detector);
    tag36h11_destroy(d->family);
    free(d);
}

int yolo_tag_detect(yolo_tag_detector *d,
                    const uint8_t *luma, int width, int height, int stride,
                    double tag_size, double fx, double fy, double cx, double cy,
                    yolo_tag_result *results, int max_results) {
    image_u8_t image = { .width = width, .height = height, .stride = stride, .buf = (uint8_t *)luma };
    zarray_t *detections = apriltag_detector_detect(d->detector, &image);
    if (!detections) return 0;

    int count = 0;
    for (int i = 0; i < zarray_size(detections) && count < max_results; i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);

        yolo_tag_result *r = &results[count++];
        r->id = det->id;
        r->hamming = det->hamming;
        r->decision_margin = det->decision_margin;
        for (int k = 0; k < 4; k++) {
            r->corners[2 * k] = det->p[k][0];
            r->corners[2 * k + 1] = det->p[k][1];
        }

        apriltag_detection_info_t info = { det, tag_size, fx, fy, cx, cy };
        apriltag_pose_t pose;
        r->pose_error = estimate_tag_pose(&info, &pose);
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) r->R[row * 3 + col] = matd_get(pose.R, row, col);
            r->t[row] = matd_get(pose.t, row, 0);
        }
        matd_destroy(pose.R);
        matd_destroy(pose.t);
    }
    apriltag_detections_destroy(detections);
    return count;
}
