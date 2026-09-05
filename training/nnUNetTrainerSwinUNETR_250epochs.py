import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR import nnUNetTrainerSwinUNETR


class nnUNetTrainerSwinUNETR_250epochs(nnUNetTrainerSwinUNETR):
    """U-Mamba deposunun resmi SwinUNETR trainer'i, 250 epoch'a sabitlenmis.
    Mimari, optimizer (AdamW 8e-4 + Cosine), AMP-kapali ve grad-clip=12
    ayarlari degistirilmedi; yalnizca butce diger kollarla esitlendi."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        self.print_to_log_file("SwinUNETR_250epochs: budget matched to other arms (250 epochs)")
