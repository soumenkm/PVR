import os, json, pickle, torch, argparse, tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from dataset import LogicDataset
from model import PVRModel
import numpy as np
from typing import List, Tuple, Union, Any

class PVRTrainer:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.model_name = self.config["model_name"]
        
        self.train_ds = LogicDataset(filepath=self.config["train_dataset_path"], model_name=self.model_name, frac=self.config["train_frac"])
        self.test_ds = LogicDataset(filepath=self.config["test_dataset_path"], model_name=self.model_name, frac=self.config["test_frac"])

        self.train_dataloader = DataLoader(self.train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(self.train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)

        self.num_epochs = self.config["num_epochs"]
        self.project_name = "PVR_train"
        os.environ["WANDB_PROJECT"] = self.project_name
        
        self.run_name = f"{self.config['train_frac']:.2f}_{self.config['initial_lr']:.1e}"
        self.model = PVRModel(d_model=1024, model_name=self.model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
   
        self.output_dir = Path(Path.cwd(), f"outputs/ckpt/{self.project_name}/{self.run_name}")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = Path(self.output_dir, f"model_epoch_{epoch}.pt")
        data = {"state_dict": self.model.state_dict(), "config": self.config}
        torch.save(data, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
    
    def _evaluate(self, epoch: int) -> None:
        epoch_loss = 0.0
        epoch_acc = 0.0
        with tqdm.tqdm(self.test_dataloader, desc=f"Eval Epoch [{epoch}/{self.num_epochs}]") as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = {key: val.to(self.device) for key, val in batch.items()}
                out = self.model(question_ids=batch["question"], rules_ids=batch["rules"], facts_ids=batch["facts"], labels=batch["answer"])
                logits = out["logits"]
                loss = out["loss"]
                
                epoch_loss += loss.item()
                acc = (logits.argmax(dim=1) == batch["answer"]).to(float).mean()
                epoch_acc += acc.item()
                
                pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}",
                                    "Batch Acc": f"{acc.item():.4f}"})
                if self.config["wandb_log"]:
                    wandb.log({"Eval Loss": loss.item(), "Eval Acc": acc.item(), "Epoch": epoch, "Step": self.num_eval_steps})
                    self.num_eval_steps += 1

        avg_loss = epoch_loss / len(self.test_dataloader)
        avg_acc = epoch_acc / len(self.test_dataloader)
        print(f"Eval Epoch [{epoch}/{self.num_epochs}] completed with Average Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    
    def train(self) -> None:
        self.model.calc_num_params()
        if self.config["wandb_log"]:
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            self.num_eval_steps = 0
            self.num_steps = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            epoch_acc = 0.0
            with tqdm.tqdm(self.train_dataloader, desc=f"Train Epoch [{epoch}/{self.num_epochs}]") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    batch = {key: val.to(self.device) for key, val in batch.items()}
                    out = self.model(question_ids=batch["question"], rules_ids=batch["rules"], facts_ids=batch["facts"], labels=batch["answer"])
                    logits = out["logits"]
                    loss = out["loss"]
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    acc = (logits.argmax(dim=1) == batch["answer"]).to(float).mean()
                    epoch_acc += acc.item()
                    
                    pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}",
                                      "Batch Acc": f"{acc.item():.4f}"})
                    if self.config["wandb_log"]:
                        wandb.log({"Train Loss": loss.item(), "Train Acc": acc.item(), "Epoch": epoch, "Step": self.num_steps})
                        self.num_steps += 1

            if epoch % 1 == 0:
                self._save_checkpoint(epoch=epoch)
                self._evaluate(epoch=epoch)
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            avg_acc = epoch_acc / len(self.train_dataloader)
            print(f"Train Epoch [{epoch}/{self.num_epochs}] completed with Average Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
        if self.config["wandb_log"]:
            wandb.finish()

    @staticmethod
    def load_model(device: torch.device, checkpoint_path: Path) -> dict:
        ckpt = torch.load(checkpoint_path, map_location=device)
        config = ckpt["config"]
        model = PVRModel(d_model=1024, model_name=config["model_name"])
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        return {"model": model, "config_data": config}
            
def main(device: torch.device) -> None:
    config = {
        "model_name": "LIAMF-USP/roberta-large-finetuned-race",
        "train_dataset_path": Path(Path.cwd(), "data/depth-5/meta-train.jsonl"),
        "test_dataset_path": Path(Path.cwd(), "data/depth-5/meta-test.jsonl"),
        "train_frac": 1.0,
        "test_frac": 0.1,
        "num_epochs": 5,
        "num_steps": None,
        "batch_size": 16,
        "initial_lr": 1e-5,
        "max_grad_norm": 1.0,
        "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999),
        "grad_acc_steps": 1,
        "num_ckpt_per_epoch": 1,
        "wandb_log": True
    }

    trainer = PVRTrainer(device=device, config=config)
    trainer.train()
    
    # out = PVRTrainer.load_model(
    #     device="cuda", 
    #     checkpoint_path=Path(Path.cwd(), "outputs/ckpt/PVR_train/0.01_5.0e-05/model_epoch_1.pt"),
    # )
    # print(out)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(device=device)