from pytorch_lightning.callbacks import Callback
import wandb
import torch
from torchmetrics.functional import calibration_error
from callbacks import ProbablisticSoftmaxPostprocessor

class ECECallback(Callback):

    def __init__(self):
        super().__init__()
        self.predictions = []
        self.targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # print("entered on validation batch end in logging callbacks")
        if trainer.current_epoch == (trainer.max_epochs-1):
            y = batch["anno"]
            print('\n',"batch anno shape",batch["anno"].shape,'\n')

            softmaxPostprocessor = ProbablisticSoftmaxPostprocessor()
            logits = outputs["logits"]
            print('\n',"preprocessing logits shape:",logits.shape,'\n')
            logits = softmaxPostprocessor.process_logits(logits)
            print('\n',"softmax shape:", logits.shape,'\n')
            self.predictions.append(logits.detach().cpu())
            self.targets.append(y.detach().cpu())
    
    def on_validation_end(self, trainer, pl_module):
        print('\n',"entered on validation end in logging callbacks"'\n')

        if trainer.current_epoch == (trainer.max_epochs-1):
            preds = torch.cat(self.predictions)
            targets = torch.cat(self.targets)
            ece = self._compute_ece(preds, targets)
            print('\n',"ECE is:", ece,'\n')
            # wandb.log({"ECE Validation Dataset": ece})

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
    
