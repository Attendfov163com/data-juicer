import numpy as np
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS
from data_juicer.utils.ASD_mapper_utils import get_video_array_cv2
import gc

OP_NAME = 'video_captioning_face_attribute_emotion_mapper'

import torch, os, tempfile, shutil
from shutil import rmtree
import pickle, copy, cv2
import transformers  # noqa: F401

# avoid hanging when calling clip in multiprocessing
torch.set_num_threads(1)
import sys


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFaceAttributeEmotionMapper(Mapper):
    _accelerator = 'cuda'
    _batched_op = True

    def __init__(
        self,
        face_track_query: str = "Please describe the person's facial expression, tell me the person's emotion through the video, like Happiness, Excitement, Love, Gratitude, Relief, Pride, Anger, Sadness, Fear, Guilt, Shame, Disgust, Surprise, Confusion, Curiosity, Boredom ...",
        trust_remote_code: bool = False,
        cropping_face_video_tempt_path = './tempt_video/tmp_video_remove',
        video_describe_model_path: str = 'DAMO-NLP-SG/VideoLLaMA3-7B',
        video_facetrack_attribute_emotion: str = MetaKeys.video_facetrack_attribute_emotion,
        *args,
        **kwargs
    ):
        """
        Initialization method.

        :param hf_video_blip: video-blip model name on huggingface
            to generate caption
        """
        kwargs.setdefault('mem_required', '40GB')
        super().__init__(*args, **kwargs)

        self._batched_op = True
        self._accelerator = 'cuda'
        self.context_param = 0.8

        # self.pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. "
        self.query = face_track_query
        self.cropping_face_video_tempt_path = cropping_face_video_tempt_path

        self.video_describe_model_path = video_describe_model_path if video_describe_model_path else 'DAMO-NLP-SG/VideoLLaMA3-7B'
        self.model_key = prepare_model(
            model_type='huggingface',
            pretrained_model_name_or_path=video_describe_model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        self.video_facetrack_attribute_emotion = video_facetrack_attribute_emotion



    def process_single(self, samples, rank=None):

        if not MetaKeys.human_track_data_path in samples[Fields.meta]:
            raise ValueError("video_captioning_from_human_tracks_mapper must be operated after video_human_tracks_extraction_mapper.")
        

        Total_information = []
        video_samples = samples[Fields.meta][MetaKeys.human_track_data_path]
        loaded_video_keys = samples[self.video_key]
        
        cropping_face_video_tempt_path = tempfile.mkdtemp(dir=self.cropping_face_video_tempt_path)
        if os.path.exists(cropping_face_video_tempt_path):
            rmtree(cropping_face_video_tempt_path)

        os.makedirs(cropping_face_video_tempt_path, exist_ok = False)
        model, processor = get_model(self.model_key, rank, self.use_cuda())
        for vedio_id,ASD_attribute_all_tracks_for_one_video in enumerate(video_samples):
            if len(ASD_attribute_all_tracks_for_one_video) == 0:
                Total_information.append([])
                continue
            
            description_for_each_track = []
            video_array = get_video_array_cv2(loaded_video_keys[vedio_id])
            for track_id,tracks_now in enumerate(ASD_attribute_all_tracks_for_one_video):
                cs = self.context_param  

                with open(tracks_now['bbox_path'], 'rb') as f:
                    bbox_data = pickle.load(f)
                    xys_bbox = bbox_data['xys_bbox']
                    track_frame = bbox_data['frame']

                face_video_out_path = os.path.join(cropping_face_video_tempt_path, loaded_video_keys[vedio_id].split('/')[-1][:-4] + '__' + str(track_id) + '.mp4')
                vOut = cv2.VideoWriter(face_video_out_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	
                start_frame_id_in = 0
                start_frame_id_out = track_frame[start_frame_id_in]  # tag
                while start_frame_id_in + 1 <len(track_frame):
                    bs = xys_bbox['s'][start_frame_id_in]  
                    bsi = int(bs * (1 + 2 * cs))  

                    start_frame_id_in = start_frame_id_in + 1
                    start_frame_id_out = track_frame[start_frame_id_in]
                    frame_before_crop = video_array[start_frame_id_out]

                    frame = np.pad(frame_before_crop, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
                    my  = xys_bbox['y'][start_frame_id_in] + bsi  # BBox center Y
                    mx  = xys_bbox['x'][start_frame_id_in] + bsi  # BBox center X
                    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
                    vOut.write(cv2.resize(face, (224, 224)))
                vOut.release()
                
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": {"video_path": face_video_out_path, "fps": 1, "max_frames": 180}},
                            {"type": "text", "text": self.query},
                        ]
                    },
                ]

                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                output_ids = model.generate(**inputs, max_new_tokens=1024)

                outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                description_for_each_track.append(outputs)
            
            Total_information.append(description_for_each_track)

        shutil.rmtree(cropping_face_video_tempt_path)
        samples[Fields.meta][self.video_facetrack_attribute_emotion] = [Total_information]
        gc.collect()
        torch.cuda.empty_cache()
        return samples