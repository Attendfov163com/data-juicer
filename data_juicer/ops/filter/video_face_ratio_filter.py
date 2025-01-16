import av
import numpy as np
from jsonargparse.typing import ClosedUnitInterval
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import (load_data_with_context, load_video,
                                        pil_to_opencv, pil_to_opencv, process_each_frame)
from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_VIDEOS

import psutil
import gc,os

OP_NAME = 'video_face_ratio_filter'

with AvailabilityChecking(['dlib', 'Pillow'], OP_NAME):
    import cv2,dlib
    from PIL import ImageFilter

@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoFaceRatioFilter(Filter):
    """Keep data samples whose videos' durations are within a specified range.
    """

    def __init__(self,
                 threshold: ClosedUnitInterval = 0.8,
                 detect_interval: int = 1,
                 any_or_all: str = 'all',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.threshold = threshold

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()
        # self.detector_key = prepare_model(model_type='face_detect_S3FD')


        self.detect_interval = detect_interval


    def compute_stats(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_face_exist in sample[Fields.stats]:
            return sample
        
        # load videos
        loaded_video_keys = sample[self.video_key]
        video_faces_ratio = {}
        
        # face_detect_S3FD = get_model(self.detector_key, rank=rank)

        process = psutil.Process(os.getpid())
        # memory_before = process.memory_info().rss / 1024 ** 2  # MB


        for video_key in loaded_video_keys:
            try:
                with av.open(video_key) as container:
                    # getting video stream
                    video_stream = next(s for s in container.streams if s.type == 'video')
                    # iterate over the video frame and detect faces
                    frame_counter = 0  
                    total_frames = 0
                    frames_with_face = 0
                    detect_num = 0
                    for packet in container.demux(video_stream):
                        try:
                            for frame in packet.decode():
                                total_frames += 1
                                frame_counter += 1  

                                if frame_counter % self.detect_interval == 0:
                                    detect_num = detect_num + 1
                                    img = frame.to_image()
                                    image = pil_to_opencv(img)
                                    # imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    # faces = face_detect_S3FD.detect_faces(imageNumpy, conf_th=0.9, scales=[0.25])
                                    faces = self.detector(image)
                                    if len(faces) > 0:
                                        frames_with_face += 1
                        except Exception as e:
                            print(f"Frame decoding error in video {video_key}: {e}")
                            frames_with_face = 0
                            detect_num = 0

                    # calculate the proportion of the number of face frames
                    if detect_num > 0:
                        face_ratio = frames_with_face / detect_num
                    else:
                        face_ratio = 0.0
                    video_faces_ratio[video_key] = face_ratio
            except av.AVError as e:
                print(f"Error opening video {video_key}: {e}")
                video_faces_ratio[video_key] = 0.0
            finally:
                container.close()

            video_faces_ratio[video_key] = face_ratio

        # get video faces ratio
        sample[Fields.stats][StatsKeys.video_face_exist] = [
            video_faces_ratio[video_key] for video_key in sample[self.video_key]
        ]

        memory_after = process.memory_info().rss / 1024 ** 2  # MB
        print(f"Memory Usage: {memory_after:.2f} MB")

        gc.collect()

        return sample

    def process(self, sample):
        video_faces_ratio = sample[Fields.stats][StatsKeys.video_face_exist]
        keep_bools = np.array([
            duration >= self.threshold
            for duration in video_faces_ratio
        ])
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
