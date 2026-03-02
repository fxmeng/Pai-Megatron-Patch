import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
image_token_id, first_vision_token_id = tokenizer(["<|image_pad|>", "<|vision_0|>"])["input_ids"]
spatial_merge_size = 2
merge_length = spatial_merge_size ** 2

def conversation_to_tokens_batch(batch):
    all_tokens = []
    all_token_lens = []

    for conversations, discrete_tokens in zip(
        batch["conversations"], batch["discrete_tokens"]
    ):
        # 拼接文本
        conversation_text = "\n".join([conv["content"] for conv in conversations])
        conversation_text = conversation_text.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
        )

        input_ids = tokenizer(
            conversation_text,
            padding="do_not_pad",
            return_tensors="np"
        ).input_ids[0]

        image_token_indices = np.where(input_ids == image_token_id)[0]
        num_discrete_tokens = sum(len(sub) for sub in discrete_tokens)


        target_length = (
            input_ids.shape[0]
            - len(discrete_tokens)
            + num_discrete_tokens // merge_length
            + num_discrete_tokens
        )

        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)

        cur_x, cur_y, image_idx = 0, 0, 0

        for i, idx in enumerate(image_token_indices):
            num_disc = len(discrete_tokens[i])
            token_id = input_ids[idx]

            size = len(discrete_tokens[image_idx]) // merge_length
            image_idx += 1

            # 文本部分
            final_input_ids[cur_y: cur_y + idx - cur_x] = input_ids[cur_x:idx]
            cur_y += idx - cur_x

            # image token 展开
            final_input_ids[cur_y: cur_y + size] = token_id
            cur_y += size
            cur_x = idx + 1

            # discrete tokens
            final_input_ids[cur_y: cur_y + num_disc] = (
                np.array(discrete_tokens[i]) + first_vision_token_id
            )
            cur_y += num_disc

        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]

        all_tokens.append(final_input_ids.tolist())
        all_token_lens.append(final_input_ids.shape[0])

    return {
        "tokens": all_tokens,
        "lens": all_token_lens,
    }


dataset = load_dataset("json", data_files=sys.argv[2], split="train")

dataset = dataset.map(
    conversation_to_tokens_batch,
    batched=True,
    batch_size=512, 
    num_proc=max(os.cpu_count() - 1, 1),
    remove_columns=["conversations", "discrete_tokens"],
    desc="Building tokens",
)

print(dataset.column_names)

dataset.to_json(
    sys.argv[3],
    orient="records",   # 一行一个样本
    lines=True,         # JSONL 格式
    force_ascii=False   # 保留中文
)