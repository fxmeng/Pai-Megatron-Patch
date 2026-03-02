import os
import sys
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

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

def iter_jsonl_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                yield os.path.join(dirpath, fn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="/workspace/data_02111332/vl_jsonl/")
    parser.add_argument("--output_root", type=str, default="/workspace/data_02111332/vl_tokens_jsonl/")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer name or local path")
    parser.add_argument("--num_proc", type=int, default=max(os.cpu_count() - 1, 1))
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    image_token_id, first_vision_token_id = tokenizer(["<|image_pad|>", "<|vision_0|>"])["input_ids"]

    jsonl_files = list(iter_jsonl_files(input_root))
    if not jsonl_files:
        print(f"No .jsonl files found under: {input_root}")
        sys.exit(0)

    print(f"Found {len(jsonl_files)} jsonl files under: {input_root}")
    print(f"Output root: {output_root}")

    for in_file in jsonl_files:
        rel_path = os.path.relpath(in_file, input_root)
        out_file = os.path.join(output_root, rel_path)

        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_file):
            print(f"Skip (already exists): {out_file}")
            continue

        print(f"\nProcessing:\n  IN : {in_file}\n  OUT: {out_file}")

        ds = load_dataset("json", data_files=in_file, split="train")
        ds = ds.map(
            lambda batch: conversation_to_tokens_batch(
                batch,
                tokenizer=tokenizer,
                image_token_id=image_token_id,
                first_vision_token_id=first_vision_token_id,
                spatial_merge_size=2,
            ),
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=["conversations", "discrete_tokens"],
            desc=f"Tokenizing {rel_path}",
        )

        # 保存为 jsonl（同结构）
        ds.to_json(
            out_file,
            orient="records",
            lines=True,
            force_ascii=False,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()