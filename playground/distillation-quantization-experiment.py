"""
Script to understand and explore distillation, and quantization using SetFit
Code is based on: https://github.com/huggingface/workshops/tree/main/fewshot-learning-in-production
"""
import os
from pathlib import Path
from time import perf_counter
from typing import Union

import datasets
import evaluate
import numpy as np
import torch
from setfit import sample_dataset, SetFitModel, SetFitTrainer, DistillationSetFitTrainer

# from tqdm.auto import tqdm


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

    def compute_size(self) -> dict:
        state_dict = self.model.model_body.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        tmp_path.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_model(self, query: str = "What is the pin number for my account?") -> dict:
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

    def run_benchmark(self) -> dict:
        metrics = {self.optim_type: self.compute_size()}
        metrics[self.optim_type].update(self.compute_accuracy())
        metrics[self.optim_type].update(self.time_model())
        return metrics


def get_dataset_splits(
        dataset: datasets.DatasetDict
) -> tuple[datasets.DatasetDict, datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    # Create 2 splits: one for few-shot training, the other for knowledge distillation
    train_data = dataset["train"].train_test_split(seed=42)
    # Sample 8 examples / class for fine-tuning
    train_data_teacher = sample_dataset(train_data["train"])
    # Select 1000 unlabeled examples for knowledge distillation
    train_data_student = train_data["test"].select(range(1000))
    # Define the test set for evaluation
    test_data = dataset["test"]
    return train_data, train_data_teacher, train_data_student, test_data


def train_model(
        model: Union[SetFitModel, str], training_dataset: datasets.Dataset
) -> SetFitTrainer:
    if isinstance(model, str):
        model = SetFitModel.from_pretrained(model)
    trainer = SetFitTrainer(model=model, train_dataset=training_dataset)
    trainer.train()
    return trainer


def train_distillation_model(
        teacher_model: SetFitModel,
        student_model: Union[str, SetFitModel],
        training_dataset: datasets.Dataset
) -> DistillationSetFitTrainer:
    if isinstance(student_model, str):
        student_model = SetFitModel.from_pretrained(student_model)
    trainer = DistillationSetFitTrainer(
        teacher_model=teacher_model,
        train_dataset=training_dataset,
        student_model=student_model,
    )
    trainer.train()
    return trainer


def get_performance_metrics(model: SetFitModel, dataset: datasets.Dataset, optim_type: str):
    pb = PerformanceBenchmark(model=model, dataset=dataset, optim_type=optim_type)
    return pb.run_benchmark()


if __name__ == '__main__':
    data = datasets.load_dataset("ag_news")
    train_dataset, train_dataset_teacher, train_dataset_student, test_dataset = get_dataset_splits(data)

    # =================================
    # TEACHER TRAINING
    # =================================

    # Load model from the Hub and train
    teacher_model_pretrained = "sentence-transformers/paraphrase-mpnet-base-v2"
    teacher_trainer = train_model(model=teacher_model_pretrained, training_dataset=train_dataset_teacher)
    perf_metrics = get_performance_metrics(teacher_trainer.model, test_dataset, "MPNet (teacher)")

    # =================================
    # STUDENT TRAINING
    # =================================
    student_model_pretrained = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    student_trainer = train_distillation_model(
        teacher_model=teacher_trainer.model,
        student_model=student_model_pretrained,
        training_dataset=train_dataset_student
    )
    perf_metrics.update(get_performance_metrics(student_trainer.student_model, test_dataset, "MiniLM-L3 (distilled)"))

    # =================================
    # TEST INCREMENTAL STUDENT TRAINING
    # =================================
    student_trainer_incr = DistillationSetFitTrainer(
        teacher_model=teacher_model,
        train_dataset=train_dataset["test"].select(range(1001, 1006)),
        student_model=student_trainer.model,
    )

    """
    Notes
    -----
    **** Teacher training ****
      Num examples = 1280
      Num epochs = 1
      Total optimization steps = 80
      Total train batch size = 16
      ---------------------------
      Training took 5:13
    
    **** Student training ****
      Num examples = 40000
      Num epochs = 1
      Total optimization steps = 2500
      Total train batch size = 16
      ---------------------------
      Training took almost 14:34
      
    **** Performance Metrics ****
    'MPNet (teacher)': 
      'size_mb': 417.7251863479614,
      'accuracy': 0.8196052631578947,
      'time_avg_ms': 76.22325751999597,
      'time_std_ms': 5.377884497735505},
    'MiniLM-L3 (distilled)': 
      'size_mb': 66.3588342666626,
      'accuracy': 0.8365789473684211,
      'time_avg_ms': 5.76854206002281,
      'time_std_ms': 1.415576433039242}
    """
