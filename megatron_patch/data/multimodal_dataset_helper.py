# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from collections import defaultdict
import numpy as np
import torch
import json

from transformers import WhisperFeatureExtractor
from megatron.training import get_args
from megatron.energon import (
    Batch,
    DefaultTaskEncoder,
    VQASample,
)

from megatron_patch.data.image_processing import get_visual_transform
from megatron_patch.data.multimodal_dataset import ChatMLSample
from megatron_patch.tokenizer import get_tokenizer


@dataclass
class EncodedSample:
    __key__: str
    __subflavors__: Dict
    
    imgs: List[np.ndarray] # (c, h, w)
    videos: List[np.ndarray] # (c, h, w)
    audios: List[np.ndarray]

    image_thw_grids: np.ndarray
    video_thw_grids: np.ndarray
    audio_lengths: np.ndarray
    image_input_mask: np.ndarray
    video_input_mask: np.ndarray
    audio_input_mask: np.ndarray
    audio_feature_attention_mask: np.ndarray
    second_per_grid_ts: np.ndarray # (n_videos, )

    text: np.ndarray
    target: np.ndarray

@dataclass
class EncodedBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (num_tiles, c, h, w)
    imgs: torch.Tensor
    videos: torch.Tensor
    audios: torch.Tensor
    image_thw_grids: torch.Tensor
    video_thw_grids: torch.Tensor
    audio_lengths: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    audio_input_mask: torch.Tensor
    audio_feature_attention_mask: torch.Tensor
    second_per_grid_ts: torch.Tensor

    # (n, seq_len)
    text: torch.Tensor
    # (n, seq_len)
    target: torch.Tensor

class InternalWarning(Warning):
    ...

def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths

def convert_to_qwen3_content(
        user_input: str, 
        image_pattern: str = '<image>',
        video_pattern: str = '<video>',
        audio_pattern: str = '<audio>'
    ):
    """
        Split user input into format Qwen2VL tokenizer accepts.
    """
    pattern = r"({image}|{video}|{audio})".format(image=image_pattern, video=video_pattern, audio=audio_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({
                "type": "text",
                "text": user_input[cur:start].strip()
            })
            
        contents.append({
            "type": matched.string[start:end][1:-1],
             matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]])
        })

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({
            "type": "text",
            "text": user_input[cur:len(user_input)].strip()
        })
    
    return contents

class TaskEncoder(DefaultTaskEncoder[ChatMLSample, EncodedSample, EncodedBatch, dict]):

    def __init__(
        self,
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()

        self.tokenizer = get_tokenizer()
        
        self.temporal_patch_size = self.args.temporal_patch_size
        self.merge_size = self.args.spatial_merge_size
        self.patch_size = self.args.patch_size

        self.seq_len = self.args.max_padding_length

        """
        WhisperFeatureExtractor extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
        Fourier Transform` which should match pytorch's `torch.stft` equivalent.
        """
        self.feature_extractor = WhisperFeatureExtractor(feature_size=128)

    def encode_sample(self, sample: ChatMLSample):
        if isinstance(sample, ChatMLSample):
            yield self.encode_chatml(sample)
        else:
            raise NotImplementedError('Sample format not supported')

    def _flatten_visual_inputs(self, visuals, is_image: bool = True):
        flattened = []
        thw_grids = []
        for visual in visuals:
            if is_image:
                resized_height, resized_width = visual.shape[-2:]
                patches = np.tile(np.array(visual), (self.temporal_patch_size, 1, 1, 1))
            else:
                assert len(visual) % self.temporal_patch_size == 0
                patches = np.array(visual)
                resized_height, resized_width = patches.shape[-2:]

            channel = patches.shape[1]
            grid_t = patches.shape[0] // self.temporal_patch_size
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)      
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
            )     
            flattened.append(flatten_patches)
            thw_grids.append((grid_t, grid_h, grid_w))
        return flattened, np.array(thw_grids)

    def encode_chatml(self, sample: ChatMLSample) -> EncodedSample:
        factor = 2 * self.args.patch_size
        min_pixels = 4 * self.args.patch_size * 4 * self.args.patch_size
        max_pixels = self.args.patch_size * self.args.patch_size * 4 * 1280
        imgs = [get_visual_transform(img, factor, min_pixels, max_pixels)[0] for img in sample.imgs]
        flattened_imgs, image_thw_grids = self._flatten_visual_inputs(imgs, is_image=True)
        conversation = json.loads(sample.conversation) if isinstance(sample.conversation, (str, bytes)) else sample.conversation
        input_ids = np.array(conversation['tokens'])
        pad_token_id = self.tokenizer.pad_token_id


        image_token_id = self.tokenizer.encode('<|image_pad|>')[0]
        image_token_indices = np.where(input_ids == image_token_id)[0]
        image_thw_grids = np.array(image_thw_grids, dtype=np.int64)
        if len(input_ids) > self.seq_len:
            raise InternalWarning(f"Long sequence with length {len(input_ids)} largger than {self.seq_len} found, dropped...")

            
        input_masks = input_ids.copy()
        input_masks[image_token_indices] = pad_token_id
        target = np.roll(input_masks, shift=-1)
        target[-1] = pad_token_id

        if (target == pad_token_id).all():
            raise InternalWarning("Sample with all masked label, dropped.")

        image_input_mask = input_ids == self.tokenizer.image_token_id
        video_input_mask = input_ids == self.tokenizer.video_token_id
        audio_input_mask = input_ids == self.tokenizer.audio_token_id

        # collect data
        return EncodedSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs,
            videos=[],
            audios=None,
            image_thw_grids=image_thw_grids,
            video_thw_grids=[],
            audio_lengths=[],
            second_per_grid_ts = np.array([], dtype=np.float32),
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            audio_input_mask=audio_input_mask,
            audio_feature_attention_mask= None,
            text=input_ids,
            target=target,
        )

    def batch(self, samples: List[EncodedSample]) -> EncodedBatch:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]
        if len(imgs) > 0:
            imgs = torch.cat([torch.from_numpy(img) for img in imgs])
        else:
            imgs = torch.empty([0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size], dtype=torch.float32)
        
        image_thw_grids = [thw_grids for s in samples for thw_grids in s.image_thw_grids]
        if len(image_thw_grids) > 0:
            image_thw_grids = torch.from_numpy(np.array(image_thw_grids)).long()
            assert image_thw_grids.prod(dim=-1).sum() == imgs.shape[0]
        else:
            image_thw_grids = torch.empty([0, 3], dtype=torch.long)
        
        # Stack videos to [num_tiles, c, h, w]. If there are no videos (text-only), then use a dummy video.
        videos = [video for s in samples for video in s.videos]
        if len(videos) > 0:
            videos = torch.cat([torch.from_numpy(video) for video in videos])
        else:
            videos = torch.empty([0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size], dtype=torch.float32)
        
        second_per_grid_ts = [second_per_grid for s in samples for second_per_grid in s.second_per_grid_ts]
        if len(second_per_grid_ts) > 0:
            second_per_grid_ts = torch.from_numpy(np.array(second_per_grid_ts)).float()
        else:
            second_per_grid_ts = torch.empty([0, ], dtype=torch.float32)
        
        video_thw_grids = [thw_grids for s in samples for thw_grids in s.video_thw_grids]
        if len(video_thw_grids) > 0:
            video_thw_grids = torch.from_numpy(np.array(video_thw_grids)).long()
            assert video_thw_grids.prod(dim=-1).sum() == videos.shape[0]
        else:
            video_thw_grids = torch.empty([0, 3], dtype=torch.long)

        audios = [s.audios for s in samples if s.audios is not None]
        if len(audios) > 0:
            audios = torch.from_numpy(np.concatenate(audios, axis=0)) # (N, D, T)
        else:
            audios = torch.empty([0, 1, 1], dtype=torch.long) # 

        audio_lengths = [s.audio_lengths for s in samples]
        if len(audio_lengths) > 0:
            audio_lengths = torch.from_numpy(np.concatenate(audio_lengths, axis=0))
        else:
            audio_lengths = torch.empty([0, ], dtype=torch.long)
 
        # If the user hasn't defined a target sequence length, then use the max along the sample lengths.
        max_seq_len = self.seq_len
        if not max_seq_len:
            max_seq_len = max(len(s.text) for s in samples)

        text_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        # +1 to accommodate shift to left by one later.
        target_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        
        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        audio_input_masks = np.zeros_like(text_mat, dtype=bool)
        audio_feature_attention_masks = np.zeros_like(text_mat, dtype=bool)
        
        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))

            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            # NOTE: we should assert user input sequence will not be truncated
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            if s.audio_input_mask is not None:
                audio_input_masks[i, :text_len] = np.array(s.audio_input_mask)[:text_len]
            if s.audio_feature_attention_mask is not None:
                audio_feature_length = s.audio_feature_attention_mask.shape[1]
                audio_feature_attention_masks[i, :audio_feature_length] = np.array(s.audio_feature_attention_mask)[:audio_feature_length]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]
        
    
        batch = EncodedBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            imgs=imgs,
            videos=videos,
            audios=audios,
            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            audio_lengths=audio_lengths,
            second_per_grid_ts=second_per_grid_ts,
            image_input_mask=torch.from_numpy(image_input_masks),    
            video_input_mask=torch.from_numpy(video_input_masks),
            audio_input_mask=torch.from_numpy(audio_input_masks),
            audio_feature_attention_mask=torch.from_numpy(audio_feature_attention_masks),
            text=torch.from_numpy(text_mat),
            target=torch.from_numpy(target_mat),
        )

        return batch

    def encode_batch(self, batch: EncodedBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw


def print_error_handler(exc: Exception, key: Optional[str], debug=False):
    if not debug and isinstance(exc, InternalWarning):
        return
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
