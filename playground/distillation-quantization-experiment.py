"""
Script to understand and explore distillation, and quantization using SetFit
Code is based on: https://github.com/huggingface/workshops/tree/main/fewshot-learning-in-production
"""

import os
import datasets
from pathlib import Path
from time import perf_counter
from setfit import sample_dataset, SetFitModel, SetFitTrainer, DistillationSetFitTrainer

import evaluate
import numpy as np
import torch
from tqdm.auto import tqdm

datasets.logging.set_verbosity_error()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Note: this won't update env if running other scripts


metric = evaluate.load("accuracy")


class PerformanceBenchmark:
    def __init__(self, model, dataset, optim_type):
        self.model = model
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        preds = self.model.predict(self.dataset["text"])
        labels = self.dataset["label"]
        accuracy = metric.compute(predictions=preds, references=labels)
        print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
        return accuracy

    def compute_size(self):
        state_dict = self.model.model_body.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        tmp_path.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_model(self, query="What is the pin number for my account?"):
        latencies = []
        # Warmup
        for _ in range(10):
            _ = self.model([query])
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.model([query])
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self):
        metrics = {self.optim_type: self.compute_size()}
        metrics[self.optim_type].update(self.compute_accuracy())
        metrics[self.optim_type].update(self.time_model())
        return metrics


if __name__ == '__main__':
    dataset = datasets.load_dataset("ag_news")

    # Create 2 splits: one for few-shot training, the other for knowledge distillation
    train_dataset = dataset["train"].train_test_split(seed=42)
    # Sample 8 examples / class for fine-tuning
    train_dataset_teacher = sample_dataset(train_dataset["train"])
    # Select 1000 unlabeled examples for knowledge distillation
    train_dataset_student = train_dataset["test"].select(range(1000))
    # Define the test set for evaluation
    test_dataset = dataset["test"]

    # Load pretrained model from the Hub
    teacher_model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # Create trainer
    teacher_trainer = SetFitTrainer(
        model=teacher_model, train_dataset=train_dataset_teacher
    )

    # Train!
    teacher_trainer.train()

    # Evaluate!
    pb = PerformanceBenchmark(
        model=teacher_trainer.model, dataset=test_dataset, optim_type="MPNet (teacher)"
    )
    perf_metrics = pb.run_benchmark()
