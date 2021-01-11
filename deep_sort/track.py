# vim: expandtab:ts=4:sw=4

import numpy as np


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
                 feature=None, waymo_id=None):
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

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
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
        TODO: The moving speed is 0 after identifying only one appearance; so the predicted box position is same as previous appearance.
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
        if self.hits == 1 and args.scheduler != 'merged':
            w = max(1.3 * w, 128)
            h = max(1.3 * h, 64)

            # decide corner positions
            min_width = cw - 0.5 * w
            min_height = ch - 0.5 * h
            max_width = cw + 0.5 * w
            max_height = ch + 0.5 * h
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
            w = max(w, 128)
            h = max(w, 64)

            # decide corner positions
            min_width = cw - 0.5 * w
            min_height = ch - 0.5 * h
            max_width = cw + 0.5 * w
            max_height = ch + 0.5 * h

            if v_ch > 3:  # enlarge the box when moving down
                max_height += max(0.4 * h, 3 * std_vh)
                min_width -= max(0.4 * w, 3 * std_vw)
            elif v_cw > 5 and v_w > 5 and v_h < 3:  # enlarge the box when moving right
                max_width += max(0.3 * w, 3 * std_vw)
                min_width -= 0.1 * w
            elif v_cw < -5 and v_w > 5 and v_h < 3:  # enlarge the box when moving left
                min_width -= max(0.3 * w, 3 * std_vw)
                max_width = min(1920, max_width)
            else:
                max_width += 0.1 * w
                min_width -= 0.1 * w
                max_height += 0.1 * h
                min_height -= 0.1 * h

        # limit the box coordinates
        if args.dataset == 'waymo':
            limit_w = 1920
            limit_h = 1280
        else:
            limit_w = 1248
            limit_h = 384

        min_width = max(min_width, 0)
        min_height = max(min_height, 0)
        max_width = min(max_width, limit_w)
        max_height = min(max_height, limit_h)

        return min_width, min_height, max_width, max_height
