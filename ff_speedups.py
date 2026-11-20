"""Shortcuts for live use, applied to FaceFusion in memory (its files are not modified).

Measured with the profiler (2026-11-18, 512 px, TensorRT): 87 ms per frame, of which
- ~14 ms: crc32 hash of the whole image, 10 times per frame, for the face cache (useless: every live frame is new).
  Replaced by a hash of a pixel sample (same image = same key).
- ~28 ms: identity signature (arcface) of the filmed face recomputed on every frame; your face does not change:
  it is recomputed at most every EMBEDDING_REFRESH seconds. It is only used for the identity weight ('one' mode).
- ~9 ms: blurred edge mask recomputed although it only depends on size, blur and padding: now cached.
- age, gender, race: never used here, no longer computed.
"""
import threading
import time
import zlib
from functools import lru_cache

EMBEDDING_REFRESH = 0.5


def apply() -> None:
    import facefusion.face_creator as face_creator
    import facefusion.face_masker as face_masker
    import facefusion.face_store as face_store

    def sample_hash(vision_frame) -> str:
        sample = vision_frame[::16, ::16]
        return f'{vision_frame.shape}-{zlib.crc32(sample.tobytes())}'

    store = {}
    lock = threading.Lock()

    def get_faces(vision_frame):
        with lock:
            return store.get(sample_hash(vision_frame))

    def set_faces(vision_frame, faces):
        key = sample_hash(vision_frame)
        with lock:
            if len(store) > 64:
                store.clear()
            store[key] = faces

    face_store.get_faces, face_store.set_faces = get_faces, set_faces  # face_creator calls them through the module

    original_embedding = face_creator.calculate_face_embedding
    cache = {'time': 0.0, 'value': None}

    def calculate_face_embedding(vision_frame, face_landmark_5):
        now = time.time()
        if cache['value'] is None or now - cache['time'] > EMBEDDING_REFRESH:
            cache['value'], cache['time'] = original_embedding(vision_frame, face_landmark_5), now
        return cache['value']

    face_creator.calculate_face_embedding = calculate_face_embedding
    face_creator.classify_face = lambda vision_frame, face_landmark_5: (None, None, None)

    original_box_mask = face_masker.create_box_mask

    @lru_cache(maxsize=16)
    def cached_box_mask(shape, blur, padding):
        import numpy
        return original_box_mask(numpy.zeros(shape, dtype=numpy.uint8), blur, padding)

    def create_box_mask(crop_vision_frame, face_mask_blur, face_mask_padding):
        return cached_box_mask(crop_vision_frame.shape, face_mask_blur, tuple(face_mask_padding)).copy()

    import importlib
    face_masker.create_box_mask = create_box_mask
    for name in ('face_swapper', 'expression_restorer', 'face_editor', 'face_enhancer'):
        module = importlib.import_module(f'facefusion.processors.modules.{name}.core')
        if hasattr(module, 'create_box_mask'):
            module.create_box_mask = create_box_mask
