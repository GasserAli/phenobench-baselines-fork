from pytorch_lightning.callbacks import Callback
import wandb
import torch
import os
import tifffile
from torchmetrics.functional import calibration_error
from callbacks import ProbablisticSoftmaxPostprocessor
from torchmetrics.classification import MulticlassCalibrationError

import numpy as np

class ECECallback(Callback):

    def __init__(self):
        super().__init__()
        self.predictions = []
        self.targets = []
        self.ece_metric= MulticlassCalibrationError(
            num_classes=3,  
            norm='l1',  
            n_bins=20 # Number of bins for calibration error
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print("entered on validation batch end in logging callbacks")
        if trainer.current_epoch == (trainer.max_epochs-1):
            y = batch["anno"]
            # print('\n',"batch anno shape",batch["anno"].shape,'\n')

            softmaxPostprocessor = ProbablisticSoftmaxPostprocessor()
            logits = outputs["logits"]
            # print('\n',"preprocessing logits shape:",logits.shape,'\n')
            logits = softmaxPostprocessor.process_logits(logits)
            # print('\n',"softmax shape:", logits.shape,'\n')

            self.ece_metric.update(logits, y)
            # self.predictions.append(logits.detach().cpu())
            # self.targets.append(y.detach().cpu())
    
    def on_validation_end(self, trainer, pl_module):
        # print('\n',"entered on validation end in logging callbacks"'\n')

        if trainer.current_epoch == (trainer.max_epochs-1):
            ece = self.ece_metric.compute()
            # ece = self._compute_ece(preds, targets)
            print('\n',"ECE is:", ece,'\n')
            wandb.log({"ECE Validation Dataset": ece})

    def _compute_ece(self, preds, targets, n_bins=15):
        return calibration_error(
            preds, 
            targets,
            n_bins=n_bins,
            norm='l1'
        )
    
class controlEval(Callback):

    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.eval()  # Uses running stats for BatchNorm
        return
    
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.train()  # Return to training mode
        return
    
class EntropyVisualizationCallback(Callback):
    def __init__(self):
        super().__init__()
        self.name = "entropy"
        self.entropyValues = []

    def calculate_entropy_image(self, softmax_output):
        """
        Calculate pixel-wise entropy from softmax probabilities.
        
        Args:
            softmax_output: Tensor of shape [B, C, H, W] with softmax probabilities
            
        Returns:
            entropy_image: Tensor of shape [B, 1, H, W] with entropy values
        """
        # Avoid log(0) by adding small epsilon
        eps = 1e-10
        softmax_output = torch.clamp(softmax_output, min=eps)
        
        # Calculate entropy: -sum(p * log(p)) across channels
        entropy = -torch.sum(softmax_output * torch.log(softmax_output), dim=1, keepdim=True)
        
        # Normalize to [0, 1] for visualization (optional)
        # entropy_normalized = entropy / torch.log(torch.tensor(softmax_output.size(1)))
        
        return entropy
    
    def save_entropy_images(self, entropy_tensor, fnames, path_to_dir):
        """
        Save batch of entropy images to disk as TIFF files
        
        Args:
            entropy_tensor: Tensor of shape [B, 1, H, W] containing entropy values
            fnames: List of filenames (length B)
            path_to_dir: Directory to save images
        """
        path_to_dir = os.path.join(path_to_dir, self.name)
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir, exist_ok=True)
        
        # Ensure tensor is on CPU and convert to numpy
        if not entropy_tensor.device == torch.device('cpu'):
            entropy_tensor = entropy_tensor.cpu()
        
        with torch.no_grad():
            entropy_images = entropy_tensor.squeeze(1).numpy().astype(np.float32)  # [B, H, W]
        
        # Save each entropy image
        for i, entropy_img in enumerate(entropy_images):
            fname = fnames[i].split('.')[0] + "_entropy.tif"  # Append _entropy to filename
            fpath = os.path.join(path_to_dir, fname)
            tifffile.imwrite(fpath, entropy_img)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.current_epoch == (trainer.max_epochs-1):
            y = batch["anno"]
            # print('\n',"batch anno shape",batch["anno"].shape,'\n')
            filenames = batch["fname"]

            path = os.path.join(trainer.log_dir, "val", "logging", f'epoch-{trainer.current_epoch:06d}')
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        
            softmaxPostprocessor = ProbablisticSoftmaxPostprocessor()
            logits = outputs["logits"]
            # print(logits.shape)
            softmax_logits = softmaxPostprocessor.process_logits(logits)
            # print(softmax_logits.shape)
            entropy = self.calculate_entropy_image(softmax_logits)
            self.save_entropy_images(entropy, filenames, path)
        return

        # return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

class ValidationLossCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        #TODO: log the validation loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            wandb.log({"val loss": val_loss})
            print(f"logged val_loss: {val_loss}")
        else:
            print("Couldn't log validation loss as val_loss is None ")
        return
    
class IoUCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        #TODO: TASK: add per class validation IoU logging 
        val_mIoU = trainer.callback_metrics.get("val_mIoU")
        if not val_mIoU:
            print(f"could not get the validation mIoU for epoch {trainer.current_epoch}")
        else:
            wandb.log({"validation mIoU": val_mIoU})
        
        if(trainer.current_epoch == trainer.max_epochs-1) and val_mIoU:
            wandb.log({"Final validation mIoU": val_mIoU})


        if trainer.current_epoch == trainer.max_epochs-1:
            wandb.define_metric(name = "Per class validation mIoU", step_metric= "class index")

            #TODO: hard coded value check if it is possible to get it from a network component  
            num_classes = 3 
            for class_idx in range(num_classes):
                iou = trainer.callback_metrics.get(f"iou_class_{class_idx}")
                # print(f"Class {class_idx} IoU: {iou:.4f}")
                wandb.log({"class index": class_idx, "Per class validation mIoU": iou})
                wandb.log({f"Class {class_idx} validation IoU": iou})
            
        return
    
    def on_train_epoch_end(self, trainer, pl_module):
        #TODO: TASK: add per class validation IoU logging 
        train_mIoU = trainer.callback_metrics.get("train_mIoU")
        if not train_mIoU:
            print(f"Could not get train mIoU for epoch {trainer.current_epoch}")
        else:
            wandb.log({"train mIoU": train_mIoU})
        
        if trainer.current_epoch == (trainer.max_epochs-1) and train_mIoU:
            wandb.log({"Final train mIoU": train_mIoU})

        if trainer.current_epoch == trainer.max_epochs-1:
            wandb.define_metric(name = "Per class training mIoU", step_metric= "class index")

            #TODO: hard coded value check if it is possible to get it from a network component  
            num_classes = 3 
            for class_idx in range(num_classes):
                iou = trainer.callback_metrics.get(f"iou_class_{class_idx}")
                print(f"Class {class_idx} IoU: {iou:.4f}")
                wandb.log({"class index": class_idx, "Per class training mIoU": iou})
                wandb.log({f"Class {class_idx} validation IoU": iou})

        return

class TrainLossCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if not train_loss:
            print(f"Could not get the train loss for epoch {trainer.current_epoch}")
        else:
            wandb.log({"Train Loss": train_loss})
        if trainer.current_epoch == trainer.max_epochs-1 and train_loss:
            wandb.log({"Final train loss": train_loss})
        return

    
