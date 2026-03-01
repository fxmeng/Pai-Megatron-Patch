
    def encode_chatml(self, sample: ChatMLSample) -> EncodedSample:
        # NOTE: generate qwen2vl conversations
        conversation = json.loads(sample.conversation) if isinstance(sample.conversation, (str, bytes)) else sample.conversation
        assert 'conversations' in conversation
        assert 'discrete_tokens' in conversation
        discrete_tokens = conversation['discrete_tokens']
        conversation = conversation['conversations']
        conversation = '\n'.join([conv['content'] for conv in conversation])
        conversation = conversation.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        input_ids = self.tokenizer(conversation, padding='do_not_pad', return_tensors="np").input_ids[0]
        pad_token_id = self.tokenizer.pad_token_id
        image_token_id, first_vision_token_id = self.tokenizer.encode(['<|image_pad|>', "<|vision_0|>"])
        image_token_indices = np.where(input_ids == image_token_id)[0]
        assert len(image_token_indices) == len(image_thw_grids), f"With {len(image_thw_grids)} images in the sample, but {len(image_token_indices)} image placeholders!"
        image_thw_grids, video_thw_grids, audio_lengths = (
            np.array(image_thw_grids, dtype=np.int64), 
            np.array(video_thw_grids, dtype=np.int64),
            np.array(audio_lengths, dtype=np.int64)
        )
        num_discrete_tokens = 0
        for sub in discrete_tokens:
            num_discrete_tokens+=len(sub)
        # (N, 3)
        target_length = (
            input_ids.shape[0] 
            - image_thw_grids.shape[0] + image_thw_grids.prod(axis=-1).sum() // merge_length
            + num_discrete_tokens
        )
        if target_length > self.seq_len:
            raise InternalWarning(f"Long sequence with length {target_length} largger than {self.seq_len} found, dropped...")
        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)
        final_input_masks = final_input_ids.copy()

        image_idx, video_idx, audio_idx = 0, 0, 0
        indices = np.sort(np.concatenate([image_token_indices, video_token_indices, audio_token_indices]))

        # WARNING: we do not implement use_audio_in_video = True
        cur_x, cur_y = 0, 0
        for i, idx in enumerate(indices):
            num_disc = len(discrete_tokens[i])
            token_id = input_ids[idx]
            if token_id == image_token_id:
                size = image_thw_grids[image_idx].prod() // merge_length
                image_idx += 1
            final_input_ids[cur_y: cur_y + idx - cur_x] = input_ids[cur_x:idx]
            cur_y += idx - cur_x
            final_input_ids[cur_y: cur_y + size] = token_id
            final_input_masks[cur_y: cur_y + size] = pad_token_id
            cur_y += size
            cur_x = idx + 1
            final_input_ids[cur_y: cur_y+num_disc] = np.array(discrete_tokens[i])+first_vision_token_id
            final_input_masks[cur_y: cur_y+num_disc] = np.array(discrete_tokens[i])+first_vision_token_id
            cur_y +=num_disc

        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]
