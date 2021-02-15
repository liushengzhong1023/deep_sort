# vim: expandtab:ts=4:sw=4

import numpy as np
from utils.phash_utils import *


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, w, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `w` is the width
    and `h` is the height.

    The track class is specific to the object to track.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, waymo_id=None, phash=None, obj_class=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.waymo_id = waymo_id

        self._n_init = n_init
        self._max_age = max_age

        # used for image similarity computation
        self.phash = phash
        self.phash_distance = 0

        # object class
        self.obj_class = obj_class

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        # ret[2] *= ret[3] # now we save the center, and w, h
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xywh())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # update the phash
        self.phash = detection.phash

    def mark_missed(self, is_key_frame=False):
        """Mark this track as missed (no association at the current time step).
        """
        # if this is the key frame delete any unmatched frames; reduce false positives in tracks
        if is_key_frame:
            self.state = TrackState.Deleted

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted

        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    # ----------------------- By Shengzhong Liu -------------------------
    def mark_deleted(self):
        '''Mark this track as dead when negative mean value appears'''
        self.state = TrackState.Deleted

    def project_next_data_region(self, args):
        '''
        Use projected state to generate a data region for next appearance.
        Goal: we want to cover the next appearance of the object.
        We utilize both mean, velocity, and covariance information.
        We guarantee that each box is a valid box on the full image.
        TODO: The moving speed is 0 after identifying only one appearance;
              so the predicted box position is same as previous appearance.
        '''
        cw, ch, w, h = self.mean[0:4]
        v_cw, v_ch, v_w, v_h = self.mean[4:8]
        std_cw, std_ch, std_vw, std_vh = np.sqrt(np.diagonal(self.covariance)[0:4])

        # filter out unqualified boxes
        flag = cw > 0 and ch > 0 and w > 0 and h > 0
        if not flag:
            self.mark_deleted()
            return [0, 0, 0, 0]

        # one appearance, no object velocity information available
        if self.hits == 1:
            w = max(1.3 * w, 96)
            h = max(1.3 * h, 64)

            # decide corner positions
            min_w = cw - 0.5 * w
            min_h = ch - 0.5 * h
            max_w = cw + 0.5 * w
            max_h = ch + 0.5 * h
        else:
            # shift the center position when the moving speed is fast
            # enter from left
            if v_cw > 5 and v_w > 5 and v_ch < 5:
                cw += v_cw

            # enter from right
            if v_cw < -5 and v_w > 5 and v_ch < 5:
                cw -= abs(v_cw)

            # move down
            if v_ch > 10:
                ch += v_ch
                cw -= 1.3 * abs(v_cw)

            # limit on box size
            w = max(w, 96)
            h = max(h, 64)

            # decide corner positions
            min_w = cw - 0.5 * w
            min_h = ch - 0.5 * h
            max_w = cw + 0.5 * w
            max_h = ch + 0.5 * h

            if v_ch > 3:  # enlarge the box when moving down
                max_h += max(0.4 * h, 3 * std_vh)
                min_w -= max(0.4 * w, 3 * std_vw)
            elif v_cw > 5 and v_w > 5 and v_h < 3:  # enlarge the box when moving right
                max_w += max(0.3 * w, 3 * std_vw)
                min_w -= 0.1 * w
            elif v_cw < -5 and v_w > 5 and v_h < 3:  # enlarge the box when moving left
                min_w -= max(0.3 * w, 3 * std_vw)
                max_w = min(1920, max_w)
            else:
                max_w += 0.1 * w
                min_w -= 0.1 * w
                max_h += 0.1 * h
                min_h -= 0.1 * h

        # limit the box coordinates
        if args.dataset == 'waymo':
            limit_w = 1920
            limit_h = 1280
        else:
            limit_w = 1248
            limit_h = 384

        min_w = int(max(min_w, 0))
        min_h = int(max(min_h, 0))
        max_w = int(min(max_w, limit_w))
        max_h = int(min(max_h, limit_h))

        # compute the Hamming distance for pHash
        if max_h - min_h > 32 and max_w - min_w > 32:
            self.hamming_distance = phash_distance(self.phash, args.preprocessed_image[min_h:max_h, min_w: max_w, :])
        else:
            self.hamming_distance = 0

        return min_w, min_h, max_w, max_h

    def project_next_data_region_v2(self, args, turn_flag=None, bg_shift=None, return_phash=False):
        '''
        Use projected state to generate a data region for next appearance.
        Goal: we want to cover the next appearance of the object.
        We utilize both mean, velocity, and covariance information.
        We guarantee that each box is a valid box on the full image.
        TODO: The moving speed is 0 after identifying only one appearance;
              so the predicted box position is same as previous appearance.
        '''
        cw, ch, w, h = self.mean[0:4]
        v_cw, v_ch, v_w, v_h = self.mean[4:8]
        # std_cw, std_ch, std_vw, std_vh = np.sqrt(np.diagonal(self.covariance)[0:4])

        # initial estimate of future position and the phash difference (deviation from the expectation)
        min_w = int(cw - w / 2)
        max_w = int(cw + w / 2)
        min_h = int(ch - h / 2)
        max_h = int(ch + h / 2)
        projected_image = args.preprocessed_image[min_h:max_h, min_w: max_w, :]
        image_h, image_w, _ = np.shape(projected_image)
        if image_h > 32 and image_w > 32:
            self.hamming_distance = phash_distance(self.phash, projected_image)
        else:
            self.hamming_distance = 0

        # box coordinate limits
        if args.dataset == 'waymo':
            limit_w = 1920
            limit_h = 1280
        else:
            limit_w = 1248
            limit_h = 384

        # process human objects
        if (args.dataset == 'waymo' and self.obj_class == 2) or (args.dataset == 'kitti' and self.obj_class == 3):
            h = max(h, 64)
            w = max(w, h, 64)
            min_w = cw - 0.6 * w
            min_h = ch - 0.6 * h
            max_w = cw + 0.6 * w
            max_h = ch + 0.6 * h

            if turn_flag == 'left_turn':
                max_w += max(0.2 * w, 50, bg_shift)
            elif turn_flag == 'right_turn':
                min_w -= max(0.2 * w, 50, bg_shift)
        else:  # vehicle class
            # avoid too small bboxes
            w = max(w, 64)
            h = max(h, 64)

            # avoid too flat bbox
            if h < w / 2:
                h = w / 2

            # one appearance, no object velocity information available
            if self.hits == 1:
                # decide corner positions
                min_w = cw - 0.7 * w
                min_h = ch - 0.7 * h
                max_w = cw + 0.7 * w
                max_h = ch + 0.7 * h

                if turn_flag == 'left_turn':
                    max_w += max(0.2 * w, 80, bg_shift)
                elif turn_flag == 'right_turn':
                    min_w -= max(0.2 * w, 80, bg_shift)
                # enter from left for the first appearance
                if min_w <= 20 and ch + h / 2 < 3 * limit_h / 4:
                    max_w += 0.5 * w
                # enter from right for the first appearance
                elif max_w >= limit_w - 20 and ch + h / 2 < 3 * limit_h / 4:
                    min_w -= 0.4 * w
                # driving from the opposite for the first appearance
                elif cw < limit_w / 3 and ch > limit_h / 2:
                    min_w -= 0.3 * w
                    max_h += 0.3 * h
            else:
                # AV left turn
                if turn_flag == 'left_turn':
                    if args.segment is not None:
                        print("left turn")
                    min_w = cw - 0.7 * w
                    min_h = ch - 0.7 * h
                    max_w = cw + 0.7 * w + max(0.2 * w, 50, bg_shift)
                    max_h = ch + 0.7 * h

                    if v_ch > 2:
                        min_w -= max(0.2 * w, 50, bg_shift)
                # AV right turn
                elif turn_flag == 'right_turn':
                    if args.segment is not None:
                        print("right turn")
                    min_w = cw - 0.7 * w - max(0.2 * w, 50, bg_shift)
                    min_h = ch - 0.7 * h
                    max_w = cw + 0.7 * w
                    max_h = ch + 0.7 * h
                # enter from left, under no turn, closer to the AV
                elif v_cw > 5 and abs(v_ch) < 5 and v_w > 5 and v_h > 0 and cw - w / 2 < 30:
                    cw += 3 * v_cw
                    min_w = cw - 0.6 * (w + 1 * v_cw + 2 * v_w)
                    max_w = cw + 0.6 * (w + 3 * v_cw + 4 * v_w)
                    min_h = ch - 0.7 * (h + abs(v_ch) + abs(v_h))
                    max_h = ch + 0.6 * (h + abs(v_ch) + abs(v_h))
                # enter from right, under no turn, further to the AV
                elif v_cw < -5 and abs(v_ch) < 5 and v_w > 5 and v_h > 0 and cw + w / 2 > limit_w - 30:
                    cw += 1.5 * v_cw
                    max_w = cw + 0.6 * (w + 1 * abs(v_cw) + 1 * v_w)
                    min_w = cw - 0.6 * (w + 2 * abs(v_cw) + 4 * v_w)
                    min_h = ch - 0.7 * (h + abs(v_ch) + abs(v_h))
                    max_h = ch + 0.6 * (h + abs(v_ch) + abs(v_h))
                # driving from the opposite
                elif v_cw < -5 and v_ch > 1 and cw < limit_w / 2:
                    cw += 1.5 * v_cw
                    ch += 1.5 * v_ch
                    min_w = cw - 0.6 * (w + 4 * abs(v_cw) + 4 * abs(v_w))
                    max_w = cw + 0.6 * (w + 3 * abs(v_cw) + 2 * abs(v_w))
                    min_h = ch - 0.6 * (h + 2 * abs(v_ch) + 0 * abs(v_h))
                    max_h = ch + 0.6 * (h + 4 * abs(v_ch) + 4 * abs(v_h))

                    if min_w < 160:
                        min_w = 0

                    if max_h > limit_h - 160:
                        max_h = limit_h
                else:
                    # decide corner positions
                    min_w = cw - 0.7 * w
                    min_h = ch - 0.7 * h
                    max_w = cw + 0.7 * w
                    max_h = ch + 0.7 * h

                    # moving very fast to the right and the tracker is too slow (the flow slicer fails)
                    if v_cw > 10:
                        max_w += v_cw * 2 + abs(v_w) * 2

                    # moving very fast to the left and the tracker is too slow (the flow slicer fails)
                    if v_cw < -10:
                        min_w -= abs(v_cw) * 2

                    # pass other objects, closer --> larger, moving left
                    if min_w < 160 and v_cw < 0:
                        min_w = 0

                    # pass other objects at right
                    if max_w > limit_w - 160 and v_cw > 0:
                        max_w = limit_w

                    # moving down
                    if max_h > limit_h - 160 and v_ch > 0:
                        max_h = limit_h

        # if args.segment is not None:
        #     print()

        min_w = int(max(min_w, 0))
        min_h = int(max(min_h, 0))
        max_w = int(min(max_w, limit_w))
        max_h = int(min(max_h, limit_h))

        if return_phash:
            return [min_w, min_h, max_w, max_h], self.hamming_distance
        else:
            return min_w, min_h, max_w, max_h

    def fake_detect(self, args):
        '''
        We use the predicted object location as the fake detection.
        We also need to handle the object age here to make sure it is not deleted too early.
        bbox: [min_w, min_h, max_w, max_h, objectness, class, class_conf]
        '''
        cw, ch, w, h = self.mean[0:4]

        # get bbox location
        min_w = cw - w / 2
        max_w = cw + w / 2
        min_h = ch - h / 2
        max_h = ch + h / 2

        # limit the box coordinates
        if args.dataset == 'waymo':
            limit_w = 1920
            limit_h = 1280
        else:
            limit_w = 1248
            limit_h = 384

        min_w = int(max(min_w, 0))
        min_h = int(max(min_h, 0))
        max_w = int(min(max_w, limit_w))
        max_h = int(min(max_h, limit_h))

        # check bbox coordinates and delete the unqualified track directly
        if max_w - min_w <= 0 or max_h - min_h <= 0:
            self.mark_deleted()
            return []

        # use the same class and fake confidence
        objectness = 1
        predicted_class = self.obj_class
        cls_conf = 1

        # update the object age
        self.time_since_update -= 1

        return [min_w, min_h, max_w, max_h, objectness, predicted_class, cls_conf]
