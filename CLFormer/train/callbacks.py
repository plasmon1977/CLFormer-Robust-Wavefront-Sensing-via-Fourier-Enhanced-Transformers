import logging


class EarlyStopping:
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0,
        verbose: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
