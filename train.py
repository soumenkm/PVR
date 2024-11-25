import os, json, pickle, torch, argparse
torch.manual_seed(42)
from pathlib import Path
import wandb
# wandb.login()
from transformers import TrainingArguments, Trainer, DefaultDataCollator, get_linear_schedule_with_warmup
from dataset import LogicDataset
from model import PVRModel
import numpy as np
import evaluate
from typing import List, Tuple, Union, Any

class PVRTrainer:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.model_name = self.config["model_name"]
        self.model_name_srt = self.model_name.split("/")[-1]

        self.train_ds = LogicDataset(filepath=self.config["train_dataset_path"], model_name=self.config["model_name"], frac=self.config["train_frac"])
        self.eval_ds = LogicDataset(filepath=self.config["test_dataset_path"], model_name=self.config["model_name"], frac=self.config["test_frac"])
        self.data_collator = DefaultDataCollator(return_tensors="pt")

        self.batch_size = self.config["batch_size"]
        self.config["num_steps"] = len(self.train_ds)//self.batch_size
        self.num_steps = self.config["num_steps"]
        self.num_epochs = self.config["num_epochs"]
        self.project_name = f"{self.model_name_srt}_train"
        os.environ["WANDB_PROJECT"] = self.project_name
        
        self.run_name = f"{self.config['initial_lr']:.1e}"
        self.model = PVRModel(d_model=1024, model_name=self.config["model_name"])
            
        self.output_dir = f"outputs/ckpt/{self.project_name}/{self.run_name}"
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=int(0.01 * self.num_steps), 
                                                         num_training_steps=int(self.num_epochs * self.num_steps))

        self.train_arg_config = {
            "output_dir": self.output_dir,
            "eval_strategy": "steps",
            "eval_steps": max(self.batch_size, self.num_steps//self.config["num_ckpt_per_epoch"]),
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.config["grad_acc_steps"],
            "max_grad_norm": self.config["max_grad_norm"],
            "num_train_epochs": self.num_epochs,
            "logging_strategy": "steps",
            "logging_first_step": True,
            "logging_steps": self.batch_size,
            "save_strategy": "steps",
            "save_steps": max(self.batch_size, self.num_steps//self.config["num_ckpt_per_epoch"]),
            "save_safetensors": False, 
            "save_total_limit": self.num_epochs * self.config["num_ckpt_per_epoch"] + 1,
            "save_only_model": False,
            "fp16": False,
            "bf16": False,
            "dataloader_drop_last": True,
            "run_name": self.run_name,
            "report_to": "wandb" if self.config["wandb_log"] else "none",
            "eval_on_start": False
        }

        self.training_args = TrainingArguments(**self.train_arg_config)
        self.trainer_config = {
            "model": self.model,
            "args": self.training_args,
            "data_collator": self.data_collator,
            "train_dataset": self.train_ds,
            "eval_dataset": self.eval_ds,
            "optimizers": (self.optimizer, self.scheduler),
            "compute_metrics": self.compute_metrics
        }
        self.trainer = Trainer(**self.trainer_config)
    
    @staticmethod
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
        predictions, labels = eval_pred
        min_len = min(len(predictions), len(labels))
        predictions = np.argmax(predictions[:min_len], axis=1)
        labels = labels[:min_len]
        accuracy = evaluate.load("accuracy")
        output = accuracy.compute(predictions=predictions, references=labels) # keys: {"accuracy"}
        return output
    
    @staticmethod
    def compute_loss(self, model: torch.nn.Module, inputs: dict, outputs: torch.tensor = None):
        """Not to be used by LoRAFineTuner.
        Go to: ~/miniconda3/envs/env_name/lib/python3.8/site-packages/transformers/trainer.py
        Modify Trainer.compute_loss() function.
        Add this code snippet just before the return statement
        """
        # Custom: Begin.
        if model.training:
            total_norm = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    param_norm = p.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            preds = outputs["logits"].detach() # outputs is already calculate before
            acc = (preds.argmax(axis=1) == inputs["labels"]).to(torch.float).mean().item()
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"accuracy": acc, "param_norm": total_norm})
        # Custom: End.
    
    def _save_config(self) -> None: 
        config_data = {
            "config": self.config,
            "output_dir": self.output_dir,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "train_arg_config": self.train_arg_config
        }
        with open(self.output_dir + "/master_config.pkl", 'wb') as f:
            pickle.dump(config_data, f)
    
    # @staticmethod
    # def load_model(device: torch.device, config_path: Path, checkpoint_name: str) -> dict:
    #     with open(config_path, 'rb') as f:
    #         config_data = pickle.load(f)
    #     config = config_data["config"]
        
    #     if config["finetune_type"] == "lora":     
    #         model = VisionModelForCLSWithLoRA(device=device, model_name=config["model_name"], num_classes=config["num_classes"], lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"], linear_names=config["lora_linear_names"])
    #     elif config["finetune_type"] == "layer":
    #         model = VisionModelForCLS(device=device, model_name=config["model_name"], num_classes=config["num_classes"])
    #     else:
    #         raise ValueError("Invalid finetune type!")
        
    #     checkpoint_path = Path(config_path.parent, checkpoint_name)
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(checkpoint, strict=False)
    #     print(f"Model loaded from checkpoint: {checkpoint_path}")
    #     return {"model": model, "config_data": config_data}
    
    def train(self) -> None:
        self._save_config()
        print(self.model)
        print(self.model.calc_num_params())
        self.trainer.train(resume_from_checkpoint=False)
    
def main(device: torch.device) -> None:
    config = {
        "model_name": "LIAMF-USP/roberta-large-finetuned-race",
        "train_dataset_path": Path(Path.cwd(), "data/depth-5/meta-train.jsonl"),
        "test_dataset_path": Path(Path.cwd(), "data/depth-5/meta-test.jsonl"),
        "num_epochs": 10,
        "num_steps": None,
        "batch_size": 16,
        "initial_lr": 5e-5,
        "max_grad_norm": 10.0,
        "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999),
        "grad_acc_steps": 1,
        "num_ckpt_per_epoch": 1,
        "wandb_log": False
    }

    trainer = PVRTrainer(device=device, config=config)
    trainer.train()
    
    # out = FineTuner.load_model(
    #     device="cuda", 
    #     config_path=Path(Path.cwd(), "Project/outputs/ckpt/dinov2-base_finetune/layer_0.10_1.0e-05/master_config.pkl"),
    #     checkpoint_name="checkpoint-100/pytorch_model.bin"
    # )
    # print(out)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(device=device)