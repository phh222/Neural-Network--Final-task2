import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR,StepLR

def get_optimizer_and_scheduler(model, epochs):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        lr_min=1e-6,
        warmup_t=5,
        warmup_lr_init=1e-4,
    )
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)


    return optimizer, scheduler
