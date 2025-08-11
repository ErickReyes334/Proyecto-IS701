import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import f1_score, precision_score, recall_score

@dataclass
class FloatLabelCollator:
    padder: DataCollatorWithPadding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [torch.tensor(f["labels"], dtype=torch.float32) for f in features]
        feat_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.padder(feat_wo_labels)
        batch["labels"] = torch.stack(labels)
        return batch

def main():
    set_seed(int(os.environ.get("SEED", 42)))
    ds = load_dataset("go_emotions")
    ge_labels = ds["train"].features["labels"].feature.names
    num_labels = len(ge_labels)
    max_train = int(os.environ.get("MAX_TRAIN", "0"))
    max_eval = int(os.environ.get("MAX_EVAL", "0"))
    if max_train > 0:
        ds["train"] = ds["train"].select(range(min(max_train, len(ds["train"]))))
    if max_eval > 0:
        ds["validation"] = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))
    model_ckpt = os.environ.get("BASE_MODEL", "microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        use_safetensors=True,
    )
    def to_multilabel_vector(label_ids):
        y = np.zeros(num_labels, dtype=np.float32)
        for lid in label_ids:
            if 0 <= lid < num_labels:
                y[lid] = 1.0
        return y
    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=128)
        enc["labels"] = [to_multilabel_vector(lbls).tolist() for lbls in batch["labels"]]
        return enc
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    base_padder = DataCollatorWithPadding(tokenizer=tokenizer)
    collator = FloatLabelCollator(padder=base_padder)
    threshold = float(os.environ.get("THRESHOLD", 0.30))
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > threshold).astype(int)
        y_true = np.asarray(labels).astype(int)
        return {
            "f1_micro":        f1_score(y_true, preds, average="micro", zero_division=0),
            "precision_micro": precision_score(y_true, preds, average="micro", zero_division=0),
            "recall_micro":    recall_score(y_true, preds, average="micro", zero_division=0),
        }
    out_dir = os.environ.get("OUTPUT_DIR", "./goemo_model")
    train_bs = int(os.environ.get("TRAIN_BS", 8))
    eval_bs = int(os.environ.get("EVAL_BS", 16))
    grad_accum = int(os.environ.get("GRAD_ACCUM", 2))
    fp16_flag = os.environ.get("FP16", "true").lower() in {"1", "true", "yes"}
    bf16_flag = os.environ.get("BF16", "false").lower() in {"1", "true", "yes"}
    common_kwargs = dict(
        output_dir=out_dir,
        learning_rate=float(os.environ.get("LR", 2e-5)),
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=float(os.environ.get("EPOCHS", 2)),
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=int(os.environ.get("NUM_WORKERS", 0)),
        fp16=fp16_flag and torch.cuda.is_available(),
        bf16=bf16_flag and torch.cuda.is_available(),
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", **common_kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **common_kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ge_labels))
    print(f"✅ Modelo y tokenizer guardados en: {out_dir}")
    print(f"   Etiquetas (n={num_labels}) guardadas en labels.txt")

if __name__ == "__main__":
    main()
