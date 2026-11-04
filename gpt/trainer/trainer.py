import torch
import torch.nn as nn
from tqdm import tqdm

from gpt.dataset.gpt_dataset import GPTDataset
from gpt.model.gpt_model import GPT
from gpt.utils.misc import get_device, log_epoch, log_model, setup_logging


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = GPTDataset(self.cfg)
        self.n_epochs = self.cfg.TRAINING.n_epochs
        self.epoch_steps_train = self.cfg.TRAINING.epoch_steps_train
        self.epoch_steps_eval = self.cfg.TRAINING.epoch_steps_eval
        self.batch_size = self.cfg.TRAINING.batch_size
        self.context_size = self.cfg.TRAINING.context_size
        self.lr = float(self.cfg.TRAINING.lr)
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.build_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self):
        self.device = get_device(self.logger)
        self.model = GPT(self.cfg)
        self.model = self.model.to(self.device)
        log_model(self.cfg, self.logger, self.model, self.batch_size, self.context_size, self.device)

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            log_epoch(
                self.logger, self.writer, epoch, self.n_epochs, train_loss, val_loss, self.experiment_dir, self.model
            )

            self.logger.info(f"Sample:\n{self.generate_text(self.cfg.GPT.inf_tokens)}")

            self.writer.flush()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(range(self.epoch_steps_train), desc=f"Training epoch {epoch+1}")

        for step in pbar:
            self.optimizer.zero_grad()
            x, y = self.dataset.get_batch("train")
            x, y = x.to(self.device), y.to(self.device)
            prediction = self.model(x)
            prediction = torch.flatten(prediction, 0, 1)
            y = torch.flatten(y, 0, -1)
            loss = self.loss_fn(prediction, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.writer.add_scalar(f"Training-Loss-Step", loss.item(), self.epoch_steps_train * epoch + step)

        return total_loss / self.epoch_steps_train

    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(range(self.epoch_steps_eval), desc=f"Validating epoch {epoch+1}")

        with torch.no_grad():
            for step in pbar:
                x, y = self.dataset.get_batch("validation")
                x, y = x.to(self.device), y.to(self.device)
                prediction = self.model(x)
                prediction = torch.flatten(prediction, 0, 1)
                y = torch.flatten(y, 0, -1)
                loss = self.loss_fn(prediction, y)
                total_loss += loss.item()
                self.writer.add_scalar(f"Evaluation-Loss-Step", loss.item(), self.epoch_steps_train * epoch + step)

        return total_loss / self.epoch_steps_eval

    def generate_text(self, max_new_tokens, start_token=0):
        self.model.eval()
        context = torch.full((1, 1), start_token, dtype=torch.long, device=self.device)
        out = self.model.generate(context, max_new_tokens)
        text = self.dataset.decode(out[0])
        return text
