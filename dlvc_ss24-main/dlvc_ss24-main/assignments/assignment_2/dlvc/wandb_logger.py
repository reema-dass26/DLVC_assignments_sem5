import torch
import wandb
import os

key = os.getenv('WANDDB_KEY')
entity = os.getenv('WANDDB_ENTITY')
project = os.getenv('WANDDB_PROJECT')

wandb.login(key=key)


class WandBLogger:
    def __init__(
        self, enabled=True, model: torch.nn.modules = None, run_name: str = None
    ) -> None:

        self.enabled = enabled

        if self.enabled:
            wandb.init(entity=entity,
                        project=project
                        # group="your run group"
                        )
            if run_name is None:
                wandb.run.name = wandb.run.id
            else:
                wandb.run.name = run_name

            if model is not None:
                self.watch(model)

    def watch(self, model, log_freq: int = 1):
        wandb.watch(model, log="all", log_freq=log_freq)

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def finish(self):
        if self.enabled:
            wandb.finish()
