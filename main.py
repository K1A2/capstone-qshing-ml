import torch
import random
import numpy as np

from preprocessor.data_preprocessor import DataPreprocessor

import logging
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

    parser.add_argument('-p', '--parallel', type=int, help='멀티 gpu 사용 여부. 0=false, 1=true', default=0)
    parser.add_argument('-lf', '--log_file', type=int, help='로그 파일 출력 여부. 0=false, 1=true', default=1)
    parser.add_argument('-po', '--port', type=int, default=2033)

    parser.add_argument('-dn', '--data_name', type=str, help='데이터 파일 이름', default='')

    parser.add_argument('-ml', '--max_length', type=int, help='tokenizer length', default=512)
    parser.add_argument('-dw', '--data_workers', type=int, help='데이터 전처리 스레드 개수', default=4)

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
