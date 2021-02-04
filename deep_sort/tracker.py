# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .detection import Detection

from utils.phash_utils import *


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.9, max_age=2, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        # save a map from waymo id to tracks
        self.waymo_id_to_track = {}

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            # print(track.track_id)
            # print("before projection:", track.to_tlwh())
            track.predict(self.kf)
            # print("after projection:", track.to_tlwh())
            # print()

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            '''
            Only use appearance information to compute distances between detections and tracks.
            '''
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            # compute distance between the query features and target features
            cost_matrix = self.metric.distance(features, targets)

            # find the closest neighbors for each query
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features, the matching cascade match in order of track ages.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # mean, covariance = self.kf.initiate(detection.to_xyah())
        mean, covariance = self.kf.initiate(detection.to_xywh())

        new_track = Track(mean, covariance, self._next_id, self.n_init, self.max_age,
                          phash=detection.phash, obj_class=detection.obj_class)

        self.tracks.append(new_track)
        self._next_id += 1

        return new_track

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # skip dead tracks
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []

        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    # --------------------------------- By Shengzhong Liu --------------------------------------

    def update_with_gt_bboxes(self, gt_bbox_list, gt_waymo_id_list):
        '''
        Update the tracker with groundtruth bounding boxes in the new frames.
        bbox: [min_width, min_height, max_width, max_height]
        We match the bboxes with the track according to their waymo_id, no IoU or feature mapping is used here.
        '''
        # convert the list of gt_bboxes to list of detections
        gt_detection_list = []
        for (gt_bbox, gt_waymo_id) in zip(gt_bbox_list, gt_waymo_id_list):
            min_width, min_height, max_width, max_height = gt_bbox
            w = max_width - min_width
            h = max_height - min_height
            conf = 1
            feature = []
            gt_detection = Detection([min_width, min_height, w, h], conf, feature, gt_waymo_id)
            gt_detection_list.append(gt_detection)

        # update matched tracks and create new track for unmatched detections
        matched_tracks = []
        for gt_detection in gt_detection_list:
            if gt_detection.waymo_id in self.waymo_id_to_track:
                exist_track = self.waymo_id_to_track[gt_detection.waymo_id]
                exist_track.update(self.kf, gt_detection)
                matched_tracks.append(exist_track)
            else:
                new_track = self._initiate_track(gt_detection)
                self.waymo_id_to_track[gt_detection.waymo_id] = new_track
                matched_tracks.append(new_track)

        # mark missed for unmatched tracks
        matched_tracks = set(matched_tracks)
        for track in self.tracks:
            if track not in matched_tracks:
                track.mark_missed()

        # only preserve live tracks
        preserved_tracks = []
        preserved_waymo_id_to_track = dict()
        for track in self.tracks:
            if not track.is_deleted():
                preserved_tracks.append(track)
                preserved_waymo_id_to_track[track.waymo_id] = track
        self.tracks = preserved_tracks
        self.waymo_id_to_track = preserved_waymo_id_to_track

    def update_with_iou(self, detected_bboxes, full_image, is_key_frame=False):
        '''
        Only use the IoU score to update the tracker status, same as SORT.
        No appearance information is used.
        detected_bbox: [x_min, y_min, x_max, y_max, objectness_score, class]
        '''
        # convert the detected_boxes to detections
        detections = []
        for bbox in detected_bboxes:
            x_min, y_min, x_max, y_max = bbox[0:4]
            w = x_max - x_min
            h = y_max - y_min
            conf = bbox[4]

            # compute the phash for the partial image
            phash = compute_phash(full_image[int(y_min):int(y_max), int(x_min):int(x_max), :])
            detection = Detection([x_min, y_min, w, h], conf, feature=None, phash=phash, obj_class=bbox[5])
            detections.append(detection)

        # match detections with tracks with iou distance only
        if len(self.tracks) == 0:
            matches = []
            unmatched_tracks = []
            unmatched_detections = list(range(len(detections)))
        else:
            matches, unmatched_tracks, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks, detections)

        # update tracks with matched detections
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed(is_key_frame=is_key_frame)

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
