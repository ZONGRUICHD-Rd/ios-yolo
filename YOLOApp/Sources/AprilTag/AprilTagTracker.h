#ifndef APRILTAG_TRACKER_H
#define APRILTAG_TRACKER_H

// Temporal stabilization of AprilTag poses:
//  * Planar pose ambiguity: when both pose candidates fit about equally well, keep the one closest
//    to the track's previous pose (as ArUco's MarkerPoseTracker does) and only switch after
//    consistent error evidence; new tracks are seeded from other tags seen in the same frame
//    (the multi-marker consensus idea of Ch'ng et al., "Resolving Marker Pose Ambiguity by Robust
//    Rotation Averaging with Clique Constraints", ICRA 2020).
//  * Jitter: One Euro filter (Casiez, Roussel, Vogel, CHI 2012) on position and on rotation
//    (quaternion slerp), with cutoffs that rise with speed so fast motion doesn't lag.

#include "AprilTagBridge.h"

typedef struct yolo_tag_tracker yolo_tag_tracker;

typedef struct {
    int id;
    /// Corners from the latest detection, pixels.
    double corners[8];
    /// Filtered pose of the tag center in AprilTag's tag frame convention (see yolo_tag_result).
    double R[9];
    double t[3];
    /// Lower error / higher error of the two candidates in the latest detection (0 = unambiguous).
    double ambiguity;
    /// Candidate used from the latest detection (0 = lowest error, 1 = the alternative).
    int candidate;
    /// Seconds since the tag was last detected; 0 when detected in this update.
    double age;
} yolo_tag_track;

yolo_tag_tracker *yolo_tag_tracker_create(void);
void yolo_tag_tracker_destroy(yolo_tag_tracker *tracker);
void yolo_tag_tracker_reset(yolo_tag_tracker *tracker);

/// One Euro parameters; position cutoff rises by beta_position Hz per m/s, rotation by beta_rotation Hz per rad/s.
void yolo_tag_tracker_set_filter(yolo_tag_tracker *tracker, double min_cutoff_hz,
                                 double beta_position, double beta_rotation);
/// How long a missed tag keeps being reported, in seconds.
void yolo_tag_tracker_set_hold_time(yolo_tag_tracker *tracker, double seconds);

/// Feeds one frame of detections taken at `timestamp` (seconds). Writes the live tracks
/// (detected now or within the hold time), sorted by ID, and returns how many were written.
int yolo_tag_tracker_update(yolo_tag_tracker *tracker, const yolo_tag_result *detections, int count,
                            double timestamp, yolo_tag_track *tracks, int max_tracks);

#endif
