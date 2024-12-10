import numpy as np
from data_juicer.utils.constant import Fields
from data_juicer.utils.availability_utils import AvailabilityChecking
from deepface import DeepFace
from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.utils.ASD_mapper_utils import get_video_array_cv2
import sys
sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/ShareGPT4Video')
import gc

OP_NAME = 'video_captioning_face_attribute_demographic_mapper'

with AvailabilityChecking(['torch', 'transformers'],
                          OP_NAME):

    import torch, os
    import pickle
    
    # avoid hanging when calling clip in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFaceAttributeDemographicMapper(Mapper):
    """Mapper to generate samples whose captions are generated based on
    a video-to-text model and sampled video frame."""

    def __init__(
        self,
        original_data_save_path = '/home/daoyuan_mm/data_juicer/HumanVBenchRecipe/dj_faceattribute',
        detect_interval: int = 5,
        *args,
        **kwargs
    ):
        """
        Initialization method.

        :param hf_video_blip: video-blip model name on huggingface
            to generate caption
        """
        super().__init__(*args, **kwargs)

        self.interval = detect_interval
        self.original_data_save_path = original_data_save_path

    def process(self, samples, rank=None, context=False):

        Total_information = []
        video_samples = samples[Fields.human_track_data_path]
        loaded_video_keys = samples[self.video_key]

        for vedio_id,ASD_attribute_all_tracks_for_one_video in enumerate(video_samples):
            if len(ASD_attribute_all_tracks_for_one_video) == 0:
                Total_information.append([])
                continue
            description_for_each_track = []
            video_array = get_video_array_cv2(loaded_video_keys[vedio_id])
            for track_id,tracks_now in enumerate(ASD_attribute_all_tracks_for_one_video):
                face_attribute_dict_with_framestamp = {}

                bbox_path = tracks_now['bbox_path']
                with open(bbox_path, 'rb') as f:
                    bbox_data = pickle.load(f)
                    xys_bbox = bbox_data['xys_bbox']
                    track_frame = bbox_data['frame']

                
                total_len = len(track_frame)
                if total_len > 75:
                    interval = int(total_len/15)
                else:
                    interval = self.interval

                
                start_frame_id_in = 0
                start_frame_id_out = track_frame[start_frame_id_in]  # tag
                cs = 0.5
                while start_frame_id_in + interval <len(track_frame):
                    '''
                    start_frame_id_in = start_frame_id_in + self.interval
                    start_frame_id_out = track_frame[start_frame_id_in]
                    frame_before_crop = video_array[start_frame_id_out]
                    x1,y1,x2,y2 = xy_bbox[start_frame_id_in]
                    face_cropping_array = crop_from_array(frame_before_crop, (int(x1),int(y1),int(x2),int(y2)))
                    '''
                    
                    start_frame_id_in = start_frame_id_in + interval
                    start_frame_id_out = track_frame[start_frame_id_in]
                    frame_before_crop = video_array[start_frame_id_out]
                    bs = xys_bbox['s'][start_frame_id_in]  #
                    bsi = int(bs * (1 + 2 * cs))  #
                    frame = np.pad(frame_before_crop, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
                    my  = xys_bbox['y'][start_frame_id_in] + bsi  # BBox center Y
                    mx  = xys_bbox['x'][start_frame_id_in] + bsi  # BBox center X
                    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

                    face_message = DeepFace.analyze(
                        img_path = face, 
                        actions = ['age', 'gender', 'race'],
                        enforce_detection = False,
                        detector_backend = 'skip'
                        )
                    face_attribute_dict_with_framestamp[start_frame_id_out] = face_message
                
                mini_dict = {}
                mini_dict['data'] = face_attribute_dict_with_framestamp
                save_path = os.path.join(self.original_data_save_path,loaded_video_keys[vedio_id].split('/')[-1][:-4]+'_'+str(track_id)+'.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(face_attribute_dict_with_framestamp, f)
                
                mini_dict['face_attribute_path'] = save_path
                description_for_each_track.append(mini_dict)
            
            # 后处理
            # 存储原始的
            # 添加一些操作存到json
            save_track_infor = []
            for track_data in description_for_each_track:
                new_track_data = {}
                new_track_data['face_attribute_path'] = track_data['face_attribute_path']
                track_attri_data = track_data['data']
                age_list = [track_attri_data[key][0]['age'] for key in track_attri_data if track_attri_data[key] != []]
                dominant_gender_list = [track_attri_data[key][0]['dominant_gender'] for key in track_attri_data if track_attri_data[key] != []]
                dominant_race_list = [track_attri_data[key][0]['dominant_race'] for key in track_attri_data if track_attri_data[key] != []]
                # emotion_list = [track_attri_data[key][0]['emotion'] for key in track_attri_data if track_attri_data[key] != []]

                # age取中位
                new_track_data['age'] = self.find_median(age_list)
                # dominant_gender_list众
                new_track_data['gender'] = self.most_frequent_element_ratio(dominant_gender_list)
                # dominant_race_list众
                new_track_data['race'] = self.most_frequent_element_ratio(dominant_race_list)
                # emotion_list累加取top3
                # new_track_data['top3emotion'] = self.top_3_emotions(emotion_list)
                
                save_track_infor.append(new_track_data)

            Total_information.append(save_track_infor)

        samples[Fields.video_facetrack_attribute_demographic] = [Total_information]
        # Fields.track_video_caption
        # 手动清理内存
        gc.collect()
        return samples

    def find_median(self, int_list):
        """
        Finds the median of a list of integers.
        
        :param int_list: List of integers.
        :return: The median of the list.
        """
        sorted_list = sorted(int_list)
        n = len(sorted_list)
        
        if n == 0:
            raise ValueError("The list is empty.")
        
        mid = n // 2

        if n % 2 == 0:
            # If even, the median is the average of the two middle numbers
            median = (sorted_list[mid - 1] + sorted_list[mid]) / 2
        else:
            # If odd, the median is the middle number
            median = sorted_list[mid]
        
        return median
    
    def most_frequent_element_ratio(self, str_list):
        """
        Finds the most frequent element in a list of strings and its proportion.

        :param str_list: List of strings.
        :return: A tuple (most_frequent_element, proportion).
        """
        if not str_list:
            raise ValueError("The list is empty.")
        
        from collections import Counter
        
        # Count the frequency of each element in the list
        element_counts = Counter(str_list)
        # Find the most common element and its count
        most_common_element, most_common_count = element_counts.most_common(1)[0]
        # Calculate the proportion
        proportion = most_common_count / len(str_list)
        
        return most_common_element, str(proportion)

    def top_3_emotions(self, emotion_list):
        """
        Calculates the average score for each emotion and returns the top 3 emotions with the highest average scores.

        :param emotion_list: List of dictionaries, each containing seven emotions and their corresponding scores.
        :return: List of tuples containing the top 3 emotions and their average scores.
        """
        if not emotion_list:
            raise ValueError("The list is empty.")
        
        # Initialize a dictionary to store the total scores and counts for each emotion
        emotion_totals = {}
        emotion_counts = {}
        
        for emotion_dict in emotion_list:
            for emotion, score in emotion_dict.items():
                if emotion in emotion_totals:
                    emotion_totals[emotion] += score
                    emotion_counts[emotion] += 1
                else:
                    emotion_totals[emotion] = score
                    emotion_counts[emotion] = 1
        
        # Calculate the average score for each emotion
        emotion_averages = {emotion: emotion_totals[emotion] / emotion_counts[emotion] for emotion in emotion_totals}
        
        # Sort the emotions by their average scores in descending order and get the top 3
        top_3 = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return top_3