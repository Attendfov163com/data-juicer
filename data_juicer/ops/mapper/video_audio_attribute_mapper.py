import librosa
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import extract_audio_from_video
from thirdparty.humanvbench_models.audio_code.wav2vec_age_gender import process_func,AgeGenderModel
from ..base_op import OPERATORS, Mapper
from data_juicer.utils.model_utils import get_model, prepare_model

NAME = 'video_audio_attribute_mapper'
CHECK_PKGS = [
    'transformers', 'transformers_stream_generator', 'einops', 'accelerate',
    'tiktoken'
]

from data_juicer.utils.model_utils import get_model, prepare_model
    


@OPERATORS.register_module(NAME)
class VideoAudioAttributeMapper(Mapper):
    """Mapper to caption a video according to its audio streams based on
    Qwen-Audio model.
    """
    _accelerator = 'cuda'
    _batched_op = True

    def __init__(self, 
                 hf_audio_mapper: str = None,
                 tag_field_name: str = MetaKeys.audio_speech_attribute,
                 *args, **kwargs):
        """
        Initialization method.

        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only captioned sample in the
            final datasets and the original sample will be removed. It's True
            in default.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs.setdefault('mem_required', '7GB')
        super().__init__(*args, **kwargs)
        self._model_sampling_rate = 16000

        self._hf_summarizer = hf_audio_mapper if hf_audio_mapper else 'audeering/wav2vec2-large-robust-24-ft-age-gender'  # noqa: E501
        self.model_key = prepare_model(
            model_type='wav2vec2_age_gender',
            pretrained_model_name_or_path=self._hf_summarizer,
        )
        self.tag_field_name = tag_field_name

    def process_single(self, sample, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []
        
        if not MetaKeys.video_audio_tags in sample[Fields.meta]:
            raise ValueError("video_audio_attribute_mapper must be operated after video_tagging_from_audio_mapper.")

        # get paths of all video(s)
        loaded_video_keys = sample[self.video_key]
        audio_tag = sample[Fields.meta][MetaKeys.video_audio_tags]

        Total_result = []
        # get models
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        for i,video in enumerate(loaded_video_keys):
            audio_tag_this = audio_tag[i]
            if not audio_tag_this == 'Speech':
                Total_result.append([])
            else:
                ys, srs, valid_indexes = extract_audio_from_video(
                    video, stream_indexes=[0])
                if len(valid_indexes) == 0:
                    # there is no valid audio streams. Skip!
                    Total_result.append([])
                    continue

                # inference
                y = ys[0]
                sr = srs[0]
                # check if it meets the sampling rate condition of the model
                if sr != self._model_sampling_rate:
                    y = librosa.resample(y,
                                        orig_sr=sr,
                                        target_sr=self._model_sampling_rate)
                    sr = self._model_sampling_rate
    
                Age_female_male_child = process_func(y, sr, processor, model, device=model.device)[0]
                Age_female_male_child_dict = {}
                Age_female_male_child_dict['Age'] = [int(Age_female_male_child[0]*100)]
                Age_female_male_child_dict['female'] = [Age_female_male_child[1]]
                Age_female_male_child_dict['male'] = [Age_female_male_child[2]]
                Age_female_male_child_dict['child'] = [Age_female_male_child[3]]
                Total_result.append([Age_female_male_child_dict])

        sample[Fields.meta][self.tag_field_name] = Total_result
        return sample
