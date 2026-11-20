"""'Animated photo' mode (LivePortrait, FaceFusion models): the camera drives pose and expressions,
and the whole source photo (head, hair, bust) is what moves.

Motion is relative to a neutral pose: the average of the first REFERENCE_FRAMES frames where a face is seen (or after a
recalibration: touch the RECALIBRATE file, face the camera, mouth closed). For a stable webcam result:
- head size and position are frozen to those of the photo (otherwise the face widens as you move closer);
- angles and expression are clamped; light temporal smoothing against jitter.

FaceFusion must be initialized with the face_editor processor (it loads the LivePortrait models).
"""
import os
import threading

import cv2
import numpy

from facefusion.face_creator import get_many_faces, get_one_face
from facefusion.face_helper import paste_back, scale_face_landmark_5, warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask
from facefusion.processors.live_portrait import create_rotation, limit_expression
from facefusion.processors.modules.face_editor import core as live_portrait

TEMPLATE, SIZE = 'arcface_128', (512, 512)
RECALIBRATE = os.path.join(os.environ.get('LIVE_DIR') or os.path.expanduser('~/scratch/deepfake-direct'), 'recalibrate')
REFERENCE_FRAMES = 15
MAX_ANGLE = 30.0
SMOOTHING = 0.5  # weight of the current frame (1 = no smoothing)


class PhotoAnimator:

    def __init__(self, photo: numpy.ndarray, canvas=(1280, 720), crop_scale: float = 1.5, mask_blur: float = 0.3,
                 follow_position: bool = False) -> None:
        self.photo, self.canvas, self.crop_scale, self.follow_position = photo, canvas, crop_scale, follow_position
        face = get_one_face(get_many_faces([photo]))
        if face is None:
            raise ValueError('no face found in the source photo')
        landmark_5 = scale_face_landmark_5(face.landmark_set.get('5/68'), crop_scale)
        crop, self.affine = warp_face_by_face_landmark_5(photo, landmark_5, TEMPLATE, SIZE)
        self.mask = create_box_mask(crop, mask_blur, (0, 0, 0, 0))
        prepared = live_portrait.prepare_crop_frame(crop)
        self.feature = live_portrait.forward_extract_feature(prepared)
        self.motion = live_portrait.forward_extract_motion(prepared)
        pitch, yaw, roll, scale, translation, expression, points = self.motion
        self.source_points = scale * (points @ create_rotation(pitch, yaw, roll).T + expression) + translation
        self.lock = threading.Lock()
        self.samples = []
        self.reference = None
        self.smoothed = None
        self.last = None
        self._prepare_canvas()

    def _prepare_canvas(self) -> None:
        width, height = self.canvas
        photo_h, photo_w = self.photo.shape[:2]
        self.fit = (round(photo_w * height / photo_h), height) if photo_w / photo_h < width / height else (width, round(photo_h * width / photo_w))
        cover = max(width / photo_w, height / photo_h)
        background = cv2.resize(self.photo, (round(photo_w * cover), round(photo_h * cover)))
        top, left = (background.shape[0] - height) // 2, (background.shape[1] - width) // 2
        background = cv2.GaussianBlur(background[top:top + height, left:left + width], (0, 0), 25)
        self.background = (background * 0.55).astype(numpy.uint8)
        self.offset = ((width - self.fit[0]) // 2, (height - self.fit[1]) // 2)

    def compose(self, image: numpy.ndarray) -> numpy.ndarray:
        canvas = self.background.copy()
        x, y = self.offset
        canvas[y:y + self.fit[1], x:x + self.fit[0]] = cv2.resize(image, self.fit, interpolation=cv2.INTER_AREA)
        return canvas

    def _driving_motion(self, frame: numpy.ndarray):
        face = get_one_face(get_many_faces([frame]))
        if face is None:
            return None
        landmark_5 = scale_face_landmark_5(face.landmark_set.get('5/68'), self.crop_scale)
        crop, _ = warp_face_by_face_landmark_5(frame, landmark_5, TEMPLATE, SIZE)
        pitch, yaw, roll, scale, translation, expression, _ = live_portrait.forward_extract_motion(live_portrait.prepare_crop_frame(crop))
        scalar = lambda value: float(numpy.asarray(value).reshape(-1)[0])
        return numpy.array([scalar(pitch), scalar(yaw), scalar(roll)]), scalar(scale), numpy.array(translation, dtype=numpy.float32), numpy.array(expression, dtype=numpy.float32)

    def _update(self, motion):
        """Reference (average of the first frames) then smoothing; returns the smoothed motion, or None during calibration."""
        with self.lock:
            if os.path.exists(RECALIBRATE):
                os.remove(RECALIBRATE)
                self.samples, self.reference, self.smoothed = [], None, None
            if self.reference is None:
                self.samples.append(motion)
                if len(self.samples) < REFERENCE_FRAMES:
                    return None
                self.reference = tuple(numpy.mean([sample[i] for sample in self.samples], axis=0) for i in range(4))
            if self.smoothed is None:
                self.smoothed = motion
            else:
                self.smoothed = tuple(SMOOTHING * current + (1 - SMOOTHING) * previous for current, previous in zip(motion, self.smoothed))
            return self.smoothed, self.reference

    def __call__(self, frame: numpy.ndarray) -> numpy.ndarray:
        motion = self._driving_motion(frame)
        state = self._update(motion) if motion is not None else None
        if state is None:  # no face or calibration in progress: the photo as is (or the last frame)
            return self.last if self.last is not None else self.compose(self.photo)
        (angles, scale, translation, expression), (ref_angles, ref_scale, ref_translation, ref_expression) = state
        src_pitch, src_yaw, src_roll, src_scale, src_translation, src_expression, src_points = self.motion
        delta = numpy.clip(angles - ref_angles, -MAX_ANGLE, MAX_ANGLE)
        scalar = lambda value: float(numpy.asarray(value).reshape(-1)[0])
        rotation = create_rotation(scalar(src_pitch) + delta[0], scalar(src_yaw) + delta[1], scalar(src_roll) + delta[2])
        new_expression = limit_expression(src_expression + (expression - ref_expression))
        new_translation = src_translation
        if self.follow_position:
            shift = (translation - ref_translation).copy()
            shift[..., 2] = 0
            new_translation = src_translation + shift
        target = src_scale * (src_points @ rotation.T + new_expression) + new_translation
        target = live_portrait.forward_stitch_motion_points(target, self.source_points)
        crop = live_portrait.normalize_crop_frame(live_portrait.forward_generate_frame(self.feature, target, self.source_points))
        result = self.compose(paste_back(self.photo, crop, self.mask, self.affine))
        self.last = result
        return result
