from enum import Enum


class OptimizerEnum(Enum):
    sgd = 'sgd'
    adam = 'adam'
    sam = 'sam'

    def __str__(self):
        return self.value
    
    
class LRSchedulerEnum(Enum):
    none = 'none'
    lambda_lr = 'lambda_lr'
    step_lr = 'step_lr'
    cos_annealing = 'cos_annealing'
    custom_annealing = 'custom_annealing'
    one_cycle = 'one_cycle'
    cycle = 'cycle'
    on_plateau = 'on_plateau'

    def __str__(self):
        return self.value
