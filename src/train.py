import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.prompts import build_prompt

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tokenizer, text, category, intent):
    messages = [{"role": "user", "content": build_prompt("simple", text)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    answer = json.dumps({"category": category, "intent": intent}) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def make_collator(pad_id):
    def collate(batch):
        longest = max(len(item["input_ids"]) for item in batch)
        ids, labels, mask = [], [], []
        for item in batch:
            pad = longest - len(item["input_ids"])
            ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            mask.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto"
    )
    config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    df = pd.read_json("data/train.jsonl", lines=True)
    if args.train_size:
        df = df.head(args.train_size)
    train = [encode(tokenizer, r.text, r.category, r.intent) for r in df.itertuples()]

    out_dir = Path("checkpoints") / args.run_name
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    result = trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    record = {**vars(args), "train_examples": len(train), "train_loss": result.training_loss}
    (out_dir / "run_config.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()