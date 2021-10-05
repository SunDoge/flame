from torch.utils.tensorboard import SummaryWriter
from flame.pytorch.utils.ranking import rank0


@rank0
class Rank0SummaryWriter(SummaryWriter):
    pass
