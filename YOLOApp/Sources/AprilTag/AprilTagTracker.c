#include "AprilTagTracker.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "common/matd.h"

#define MAX_TRACKS 64
/// Without a board, switch to the other candidate once its error has been this many times lower on
/// (smoothed) average: low enough to follow the long-run better fit, high enough to stop per-frame flips.
#define SWITCH_ERROR_RATIO 1.15
#define EVIDENCE_GAIN 0.3
#define DERIVATIVE_CUTOFF_HZ 1.0
/// A candidate rotation this far from the filtered one is a re-decision, not motion: snap instead of slerp.
#define SNAP_ANGLE (25.0 * M_PI / 180.0)
/// Board plane: tags count as on it when a candidate normal is within this angle of the plane normal.
#define BOARD_AGREEMENT_ANGLE (20.0 * M_PI / 180.0)
#define BOARD_MIN_AGREEING_FRACTION 0.75
/// Time constant of the board normal's smoothing, seconds.
#define BOARD_NORMAL_TAU 0.1

typedef struct {
    int active;
    int id;
    double last_seen;
    double t_hat[3];
    double v_hat[3];
    double q_hat[4];
    double w_hat;
    /// Smoothed log(error of kept candidate / error of the other one); positive means the kept one fits worse.
    double evidence;
    double corners[8];
    double ambiguity;
    int candidate;
} track_t;

struct yolo_tag_tracker {
    track_t tracks[MAX_TRACKS];
    /// Smoothed normal (into the board) of the plane through the tag centers, when tags share one.
    double board_normal[3];
    double board_seen;
    int has_board;
    double min_cutoff;
    double beta_position;
    double beta_rotation;
    double hold_time;
};

// MARK: - Quaternions (w, x, y, z)

static void quat_from_matrix(const double *R, double *q) {
    double trace = R[0] + R[4] + R[8];
    if (trace > 0) {
        double s = sqrt(trace + 1.0) * 2;
        q[0] = 0.25 * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        double s = sqrt(1.0 + R[0] - R[4] - R[8]) * 2;
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25 * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        double s = sqrt(1.0 + R[4] - R[0] - R[8]) * 2;
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25 * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        double s = sqrt(1.0 + R[8] - R[0] - R[4]) * 2;
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25 * s;
    }
    double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; i++) q[i] /= n;
}

static void quat_to_matrix(const double *q, double *R) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w);     R[2] = 2 * (x * z + y * w);
    R[3] = 2 * (x * y + z * w);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w);     R[7] = 2 * (y * z + x * w);     R[8] = 1 - 2 * (x * x + y * y);
}

static double quat_dot(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/// Rotation angle between two orientations, radians.
static double quat_angle(const double *a, const double *b) {
    double d = fabs(quat_dot(a, b));
    return 2 * acos(d > 1 ? 1 : d);
}

/// Spherical interpolation from a toward b by fraction s, written into out (may alias a).
static void quat_slerp(const double *a, const double *b, double s, double *out) {
    double target[4];
    double d = quat_dot(a, b);
    for (int i = 0; i < 4; i++) target[i] = d < 0 ? -b[i] : b[i];
    d = fabs(d);
    double wa, wb;
    if (d > 0.9995) {
        wa = 1 - s;
        wb = s;
    } else {
        double theta = acos(d);
        wa = sin((1 - s) * theta) / sin(theta);
        wb = sin(s * theta) / sin(theta);
    }
    double result[4], n = 0;
    for (int i = 0; i < 4; i++) {
        result[i] = wa * a[i] + wb * target[i];
        n += result[i] * result[i];
    }
    n = sqrt(n);
    for (int i = 0; i < 4; i++) out[i] = result[i] / n;
}

// MARK: - One Euro filter

/// Smoothing factor of a first-order low-pass with the given cutoff, sampled every dt seconds.
static double one_euro_alpha(double cutoff_hz, double dt) {
    double tau = 1.0 / (2 * M_PI * cutoff_hz);
    return 1.0 / (1.0 + tau / dt);
}

static void filter_pose(const yolo_tag_tracker *tracker, track_t *track, const double *t, const double *q, double dt) {
    double a_d = one_euro_alpha(DERIVATIVE_CUTOFF_HZ, dt);

    double speed = 0;
    for (int i = 0; i < 3; i++) {
        double velocity = (t[i] - track->t_hat[i]) / dt;
        track->v_hat[i] += a_d * (velocity - track->v_hat[i]);
        speed += track->v_hat[i] * track->v_hat[i];
    }
    double a_t = one_euro_alpha(tracker->min_cutoff + tracker->beta_position * sqrt(speed), dt);
    for (int i = 0; i < 3; i++) track->t_hat[i] += a_t * (t[i] - track->t_hat[i]);

    double angular_speed = quat_angle(track->q_hat, q) / dt;
    track->w_hat += a_d * (angular_speed - track->w_hat);
    double a_q = one_euro_alpha(tracker->min_cutoff + tracker->beta_rotation * track->w_hat, dt);
    quat_slerp(track->q_hat, q, a_q, track->q_hat);
}

// MARK: - Tracker

yolo_tag_tracker *yolo_tag_tracker_create(void) {
    yolo_tag_tracker *tracker = calloc(1, sizeof(yolo_tag_tracker));
    tracker->min_cutoff = 1.0;
    tracker->beta_position = 8.0;
    tracker->beta_rotation = 1.5;
    tracker->hold_time = 0.3;
    return tracker;
}

void yolo_tag_tracker_destroy(yolo_tag_tracker *tracker) {
    free(tracker);
}

void yolo_tag_tracker_reset(yolo_tag_tracker *tracker) {
    memset(tracker->tracks, 0, sizeof(tracker->tracks));
}

void yolo_tag_tracker_set_filter(yolo_tag_tracker *tracker, double min_cutoff_hz,
                                 double beta_position, double beta_rotation) {
    tracker->min_cutoff = min_cutoff_hz > 1e-3 ? min_cutoff_hz : 1e-3;
    tracker->beta_position = beta_position;
    tracker->beta_rotation = beta_rotation;
}

void yolo_tag_tracker_set_hold_time(yolo_tag_tracker *tracker, double seconds) {
    tracker->hold_time = seconds;
}

static double ambiguity_of(const yolo_tag_result *d) {
    return d->candidate_count < 2 || d->error[1] <= 0 ? 0 : d->error[0] / d->error[1];
}

static track_t *find_track(yolo_tag_tracker *tracker, int id) {
    for (int i = 0; i < MAX_TRACKS; i++) {
        if (tracker->tracks[i].active && tracker->tracks[i].id == id) return &tracker->tracks[i];
    }
    return NULL;
}

static track_t *new_track(yolo_tag_tracker *tracker, int id) {
    track_t *oldest = &tracker->tracks[0];
    for (int i = 0; i < MAX_TRACKS; i++) {
        track_t *track = &tracker->tracks[i];
        if (!track->active) {
            oldest = track;
            break;
        }
        if (track->last_seen < oldest->last_seen) oldest = track;
    }
    memset(oldest, 0, sizeof(*oldest));
    oldest->active = 1;
    oldest->id = id;
    return oldest;
}

static double vec_angle(const double *a, const double *b) {
    double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return acos(d > 1 ? 1 : (d < -1 ? -1 : d));
}

/// Normal (column 2) of a row-major rotation: the tag's z axis, pointing into the tag.
static void normal_of(const double *R, double *n) {
    n[0] = R[2];
    n[1] = R[5];
    n[2] = R[8];
}

/// Fits a plane through the tag centers (positions are not affected by the rotation ambiguity) and,
/// if the tags agree they lie on it, folds its normal into the smoothed board normal.
/// Returns 1 when a board normal is available for this frame.
static int update_board(yolo_tag_tracker *tracker, const yolo_tag_result *detections, int count, double timestamp) {
    if (tracker->has_board && timestamp - tracker->board_seen > tracker->hold_time) tracker->has_board = 0;
    if (count < 3) return tracker->has_board;

    double mean[3] = { 0 };
    for (int i = 0; i < count; i++) {
        for (int c = 0; c < 3; c++) mean[c] += detections[i].t[0][c] / count;
    }
    matd_t *A = matd_create(count, 3);
    for (int i = 0; i < count; i++) {
        for (int c = 0; c < 3; c++) MATD_EL(A, i, c) = detections[i].t[0][c] - mean[c];
    }
    matd_svd_t svd = matd_svd(A);
    double spread = MATD_EL(svd.S, 0, 0), width = MATD_EL(svd.S, 1, 1);
    double normal[3] = { MATD_EL(svd.V, 0, 2), MATD_EL(svd.V, 1, 2), MATD_EL(svd.V, 2, 2) };
    matd_destroy(svd.U);
    matd_destroy(svd.S);
    matd_destroy(svd.V);
    matd_destroy(A);

    // Collinear centers don't define a plane.
    if (spread <= 0 || width < 0.2 * spread) return tracker->has_board;
    // Orient like the tags' z axes: away from the camera.
    if (normal[0] * mean[0] + normal[1] * mean[1] + normal[2] * mean[2] < 0) {
        for (int c = 0; c < 3; c++) normal[c] = -normal[c];
    }

    int agreeing = 0;
    for (int i = 0; i < count; i++) {
        for (int k = 0; k < detections[i].candidate_count; k++) {
            double n[3];
            normal_of(detections[i].R[k], n);
            if (vec_angle(n, normal) < BOARD_AGREEMENT_ANGLE) {
                agreeing++;
                break;
            }
        }
    }
    if (agreeing < BOARD_MIN_AGREEING_FRACTION * count) return tracker->has_board;

    if (!tracker->has_board || vec_angle(normal, tracker->board_normal) > 2 * BOARD_AGREEMENT_ANGLE) {
        memcpy(tracker->board_normal, normal, sizeof(normal));
    } else {
        double dt = fmax(timestamp - tracker->board_seen, 1e-3);
        double a = 1 - exp(-dt / BOARD_NORMAL_TAU), norm = 0;
        for (int c = 0; c < 3; c++) {
            tracker->board_normal[c] += a * (normal[c] - tracker->board_normal[c]);
            norm += tracker->board_normal[c] * tracker->board_normal[c];
        }
        for (int c = 0; c < 3; c++) tracker->board_normal[c] /= sqrt(norm);
    }
    tracker->has_board = 1;
    tracker->board_seen = timestamp;
    return 1;
}

/// Index of the candidate to use for this detection.
static int choose_candidate(const yolo_tag_tracker *tracker, int use_board, track_t *track, int is_continuing,
                            const yolo_tag_result *d, double q[2][4]) {
    if (d->candidate_count < 2) {
        track->evidence = 0;
        return 0;
    }
    if (use_board) {
        // The candidate whose normal matches the board plane, if it is clearly on the board.
        double n0[3], n1[3];
        normal_of(d->R[0], n0);
        normal_of(d->R[1], n1);
        double a0 = vec_angle(n0, tracker->board_normal), a1 = vec_angle(n1, tracker->board_normal);
        if (fmin(a0, a1) < BOARD_AGREEMENT_ANGLE) {
            track->evidence = 0;
            return a0 <= a1 ? 0 : 1;
        }
    }
    if (is_continuing) {
        // Keep the candidate nearest the previous pose; accumulate evidence in case that is the wrong one.
        int keep = quat_angle(q[0], track->q_hat) <= quat_angle(q[1], track->q_hat) ? 0 : 1;
        double kept_error = fmax(d->error[keep], 1e-15), other_error = fmax(d->error[1 - keep], 1e-15);
        track->evidence += EVIDENCE_GAIN * (log(kept_error / other_error) - track->evidence);
        if (track->evidence > log(SWITCH_ERROR_RATIO)) {
            keep = 1 - keep;
            track->evidence = 0;
        }
        return keep;
    }
    return 0;
}

static int compare_tracks(const void *a, const void *b) {
    return ((const yolo_tag_track *)a)->id - ((const yolo_tag_track *)b)->id;
}

int yolo_tag_tracker_update(yolo_tag_tracker *tracker, const yolo_tag_result *detections, int count,
                            double timestamp, yolo_tag_track *tracks, int max_tracks) {
    int use_board = update_board(tracker, detections, count, timestamp);

    for (int i = 0; i < count; i++) {
        const yolo_tag_result *d = &detections[i];
        double q[2][4];
        for (int k = 0; k < d->candidate_count; k++) quat_from_matrix(d->R[k], q[k]);

        track_t *track = find_track(tracker, d->id);
        int is_continuing = track && timestamp - track->last_seen <= tracker->hold_time;
        if (!track) track = new_track(tracker, d->id);

        int k = choose_candidate(tracker, use_board, track, is_continuing, d, q);
        if (is_continuing && quat_angle(q[k], track->q_hat) < SNAP_ANGLE) {
            filter_pose(tracker, track, d->t[k], q[k], fmax(timestamp - track->last_seen, 1e-3));
        } else {
            memcpy(track->t_hat, d->t[k], sizeof(track->t_hat));
            memcpy(track->q_hat, q[k], sizeof(track->q_hat));
            memset(track->v_hat, 0, sizeof(track->v_hat));
            track->w_hat = 0;
            track->evidence = 0;
        }
        track->last_seen = timestamp;
        track->ambiguity = ambiguity_of(d);
        track->candidate = k;
        memcpy(track->corners, d->corners, sizeof(track->corners));
    }

    int written = 0;
    for (int i = 0; i < MAX_TRACKS; i++) {
        track_t *track = &tracker->tracks[i];
        if (!track->active) continue;
        if (timestamp - track->last_seen > tracker->hold_time) {
            track->active = 0;
            continue;
        }
        if (written == max_tracks) continue;
        yolo_tag_track *out = &tracks[written++];
        out->id = track->id;
        memcpy(out->corners, track->corners, sizeof(out->corners));
        quat_to_matrix(track->q_hat, out->R);
        memcpy(out->t, track->t_hat, sizeof(out->t));
        out->ambiguity = track->ambiguity;
        out->candidate = track->candidate;
        out->age = timestamp - track->last_seen;
    }
    qsort(tracks, written, sizeof(yolo_tag_track), compare_tracks);
    return written;
}
