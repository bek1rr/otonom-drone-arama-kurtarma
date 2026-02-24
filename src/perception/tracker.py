import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.hits = 1
        self.age = 0
        self.missed = 0
        self.confidence = 0.0
        self.class_name = None

class ObjectTracker:
    def __init__(self, iou_threshold=0.3, max_missed=10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 1

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections):
        """
        detections: list of detection obj
        returns: updated tracks
        """

        if len(self.tracks) == 0:
            for det in detections:
                track = Track(det.bbox, self.next_id)
                track.confidence = det.confidence
                track.class_name = det.class_name
                self.tracks.append(track)
                self.next_id += 1
            return self.tracks

        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                cost_matrix[t, d] = 1 - self.iou(track.bbox, det.bbox)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - self.iou_threshold):
                self.tracks[r].bbox = detections[c].bbox
                self.tracks[r].hits += 1
                self.tracks[r].missed = 0
                self.tracks[r].confidence = detections[c].confidence
                self.tracks[r].class_name = detections[c].class_name
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Yeni trackler
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                track = Track(det.bbox, self.next_id)
                track.confidence = det.confidence
                track.class_name = det.class_name
                self.tracks.append(track)
                self.next_id += 1

        # Missed trackleri güncelle
        for i, track in enumerate(self.tracks):
            if i not in assigned_tracks:
                track.missed += 1

        # Eski trackleri sil
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

        return self.tracks