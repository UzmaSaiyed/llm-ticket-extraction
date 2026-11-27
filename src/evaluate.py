import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import score
from src.prompts import build_prompt


def generate(model, tokenizer, prompts, batch_size, max_new_tokens=48):
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", choices=["simple", "engineered"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final", action="store_true", help="required to run on test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        parser.error("The test set is only for final results. Add --final to run it.")

    df = pd.read_json(f"data/{args.split}.jsonl", lines=True)
    if args.limit:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    prompts = [build_prompt(args.prompt, text) for text in df["text"]]
    predictions = generate(model, tokenizer, prompts, args.batch_size)
    results = score(predictions, df[["category", "intent"]].to_dict("records"))
    print(json.dumps(results, indent=2))

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    name = f"{args.model.split('/')[-1]}_{args.prompt}_{args.split}"
    df.assign(prediction=predictions).to_json(
        out_dir / f"{name}_predictions.jsonl", orient="records", lines=True, force_ascii=False
    )
    (out_dir / f"{name}_scores.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()