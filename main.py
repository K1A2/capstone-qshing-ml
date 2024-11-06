import torch
import random
import numpy as np

from preprocessor.data_preprocessor import DataPreprocessor
from train import Trainer
from model.qbert import QsingBertModel

from enums import OptimizerEnum, LRSchedulerEnum

import argparse
import os

from datetime import datetime

from utils import get_default_device
from logger.main_logger import MainLogger


seed = 3243
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-pm', '--model_path', type=str, help='모델 폴더 이름', default='')
    parser.add_argument('--amp', type=int, help='amp 옵션', default=0)
    
    parser.add_argument('-p', '--parallel', type=int, help='멀티 gpu 사용 여부. 0=false, 1=true', default=0)
    parser.add_argument('-lf', '--log_file', type=int, help='로그 파일 출력 여부. 0=false, 1=true', default=1)
    parser.add_argument('-po', '--port', type=int, default=2033)
    
    parser.add_argument('-op', '--optimizer', type=OptimizerEnum, help='옵티마이저', choices=list(OptimizerEnum), default=OptimizerEnum.sgd)
    parser.add_argument('-ls', '--lr_scheduler', type=LRSchedulerEnum, help='lr 스케쥴러', choices=list(LRSchedulerEnum), default=LRSchedulerEnum.step_lr)

    parser.add_argument('-dn', '--data_name', type=str, help='데이터 파일 이름', default='')
    parser.add_argument('-sd', '--save_data', type=int, help='전처리가 완료된 데이터 파일 불러오기 - 0 false, 1 true', default=0)
    parser.add_argument('-ml', '--max_length', type=int, help='tokenizer length', default=512)
    parser.add_argument('-dw', '--data_workers', type=int, help='데이터 전처리 스레드 개수', default=4)
    parser.add_argument('-ds', '--split_ratio', type=float, help='train/validation 분할 비율', default=0.2)
    parser.add_argument('-lw', '--loader_worker', type=int, help='dataloader worker 개수', default=0)
    parser.add_argument('-b', '--batch_size', type=int, help='학습 배치사이즈', default=128)
    
    parser.add_argument('-e', '--epoch', type=int, help='epoch', default=100)
    parser.add_argument('-mlr', '--max_learning_rate', type=float, help='optimizer/scheduler max learning rate 설정 (custom cos scheduler는 반대)', default=0.1)
    parser.add_argument('-milr', '--min_learning_rate', type=float, help='optimizer/scheduler min learning rate 설정 (custom cos scheduler는 반대)', default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, help='optimizer weight decay 설정', default=5e-4)
    parser.add_argument('-gc', '--gradient_clip', type=float, help='gradient clip 설정. -1은 비활성화', default=-1)
    parser.add_argument('-lsm', '--label_smoothing', type=float, help='label smoothing 설정', default=0.0)
    parser.add_argument('-es', '--early_stopping', type=int, help='ealry stoppin epoch 지정. -1은 비활성화', default=-1)
    parser.add_argument('-snt', '--nesterov', type=int, help="nesterov sgd 사용 여부", default=1)
    parser.add_argument('--rho', type=int, help="SAM rho 파라미터", default=2.0)
    parser.add_argument('-cm', '--cos_max', type=int, help="cos annealing 주기", default=50)
    parser.add_argument('-sm', '--step_milestone', nargs='+', type=int, help='step lr scheduler milestone', default=[50])
    

    args = parser.parse_args()

    check_device = get_default_device()
    if check_device.type == 'cpu':
        args.parallel = 0

    return args


def init_distributed_training(rank, opts):
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)

    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://127.0.0.1:' + str(opts.port),
                                         world_size=opts.ngpus_per_node,
                                         rank=opts.rank)

    torch.distributed.barrier()

    def setup_for_distributed(is_master):
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    setup_for_distributed(opts.rank == 0)
    print('opts :', opts)


def main(rank: int,
         args: argparse.Namespace):
    if args.parallel == 1:
        init_distributed_training(rank, args)

    logger = MainLogger(args)
    logger.debug(f'args: {vars(args)}')

    logger.debug(f'init data preprocessing')
    data_prep = DataPreprocessor(args)
    
    model = QsingBertModel()
    trainer = Trainer(args, model, data_prep)
    trainer.train()


if __name__ == '__main__':
    args = get_arg_parse()

    if args.parallel == 1:
        args.ngpus_per_node = torch.cuda.device_count()
        args.gpu_ids = list(range(args.ngpus_per_node))
        args.num_worker = args.ngpus_per_node * 4
        args.batch_size = int(args.batch_size / args.ngpus_per_node)

        torch.multiprocessing.spawn(main,
                                    args=(args,),
                                    nprocs=args.ngpus_per_node,
                                    join=True)
    else:
        main(0, args)
