import torch

from gpt.dataset.gpt_dataset import GPTDataset
from gpt.model.gpt_model import GPT
from gpt.utils.misc import get_device, setup_logging


class Inferer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_dict_path = self.cfg.INFERENCE.state_dict_path
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.dataset = GPTDataset(self.cfg)
        self.build_model()

    def build_model(self):
        self.device = get_device(self.logger)
        self.model = GPT(self.cfg)
        self.model.load_state_dict(torch.load(self.state_dict_path))
        self.model = self.model.to(self.device)

    def inference(self, max_new_tokens, start_token=0):
        self.model.eval()
        context = torch.full((1, 1), start_token, dtype=torch.long, device=self.device)
        out = self.model.generate(context, max_new_tokens)
        text = self.dataset.decode(out[0])
        print(f"Inference:\n{text}")
