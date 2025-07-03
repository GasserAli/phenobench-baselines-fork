from pytorch_lightning.callbacks import Callback
import wandb
import torch
from torchmetrics.functional.classification import calibration_error
from callbacks import ProbablisticSoftmaxPostprocessor

class ECECallback(Callback):

    def __init__(self):
        super().__init__()
        self.predictions = []
        self.targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # print("entered on validation batch end in logging callbacks")
        if trainer.current_epoch == (trainer.max_epochs-1):
            _, y = batch["anno"]
            print("batch anno shape",batch["anno"].shape,'\n')

            softmaxPostprocessor = ProbablisticSoftmaxPostprocessor()
            logits = outputs["logits"]
            print("preprocessing logits shape:",logits.shape)
            logits = softmaxPostprocessor.process_logits(logits)
            print("softmax shape:", logits.shape,'\n')
            self.predictions.append(logits.detach().cpu())
            self.targets.append(y.detach().cpu())
    
    def on_validation_end(self, trainer, pl_module):
        print("entered on validation end in logging callbacks")

        if trainer.current_epoch == (trainer.max_epochs-1):
            preds = torch.cat(self.predictions)
            targets = torch.cat(self.targets)
            ece = self._compute_ece(preds, targets)
            print("ECE is:", ece,'\n')
            # wandb.log({"ECE Validation Dataset": ece})

    def _compute_ece(self, preds, targets, n_bins=15):
        return calibration_error(
            preds, 
            targets,
            task='multiclass',
            num_classes=3,
            n_bins=n_bins,
            norm='l1'
        )
    
    


        
