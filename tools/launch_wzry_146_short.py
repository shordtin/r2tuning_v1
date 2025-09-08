# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import argparse
from datetime import timedelta

import nncore
from nncore.engine import Engine, comm, set_random_seed
from nncore.nn import build_model

import logging
import re

def setup_eval_capture(logger: logging.Logger):
    """
    给指定 logger 挂上 EvalCaptureHandler，返回一个用于存储结果的字典。
    用法示例：
        eval_results = setup_eval_capture(logger)
        engine.launch(eval=...)
        print_eval_results(eval_results)
    """
    eval_results = {}

    class EvalCaptureHandler(logging.Handler):
        _RE = re.compile(r'Evaluation results on (\w+) split: (.+)')
        def emit(self, record):
            m = self._RE.match(record.getMessage())
            if not m:
                return
            split, body = m.groups()
            metrics = {}
            for kv in body.split(','):
                k, v = kv.split(':', 1)
                metrics[k.strip()] = float(v)
            eval_results[split] = metrics

    handler = EvalCaptureHandler()
    logger.addHandler(handler)
    return eval_results


def print_eval_results(eval_results: dict, sort_by_value: bool = False):
    """
    格式化打印 eval_results：
      - sort_by_value=False 按 metric 名字字母序
      - sort_by_value=True  按 metric 值从大到小
    """
    if not eval_results:
        print("No evaluation results were captured.")
        return

    for split, metrics in eval_results.items():
        print(f"\n===== Results for split: {split} =====")
        items = (
            sorted(metrics.items(), key=lambda x: x[1], reverse=True)
            if sort_by_value
            else sorted(metrics.items(), key=lambda x: x[0])
        )
        item_names = ['HL-min-VeryGood', 'MR-full-R1', 'MR-full-R5', 'MR-full-mAP', 'MR-full-mIoU']
        for name, val in items:
            if any(name.startswith(item) for item in item_names):
                print(f"{name:30s}: {val:6.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--checkpoint', help='load a checkpoint')
    parser.add_argument('--resume', help='resume from a checkpoint')
    parser.add_argument('--work_dir', help='working directory')
    parser.add_argument('--eval', help='evaluation mode', action='store_true')
    parser.add_argument('--eval_type', help='选择评估 "test" or "val" 数据集', choices=['val', 'test', 'both'], default='test')
    parser.add_argument('--dump', help='dump inference outputs', action='store_true')  # 将输出结果保存
    parser.add_argument('--seed', help='random seed', type=int)
    parser.add_argument('--amp', help='amp data type', type=str, default='None')       # 混合精度训练 fp16
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--launcher', help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    launcher = comm.init_dist(launcher=args.launcher, timeout=timedelta(hours=1))

    if comm.is_main_process() and not args.eval and not args.dump:
        if args.work_dir is None:
            work_dir = nncore.join('work_dirs', nncore.pure_name(args.config))
            work_dir = nncore.mkdir(work_dir, modify_path=True)
        else:
            work_dir = args.work_dir

        time_stp = nncore.get_timestamp()
        log_file = nncore.join(work_dir, '{}.log'.format(time_stp))
    else:
        log_file = work_dir = None

    logger = nncore.get_logger(log_file=log_file)
    logger.info(f'Environment info:\n{nncore.collect_env_info()}')
    logger.info(f'Launcher: {launcher}')
    logger.info(f'Config: {cfg.text}')

    seed = args.seed if args.seed is not None else cfg.get('seed')
    seed = set_random_seed(seed, deterministic=True)
    logger.info(f'Using random seed: {seed}')

    model = build_model(cfg.model, dist=bool(launcher))
    logger.info(f'Model architecture:\n{model.module}')

    params = []
    logger.info('Learnable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'{name} - {param.shape}')
            params.append(param)

    train_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = round(train_params / total_params * 100, 3)
    param = round(train_params / 1024 / 1024, 3)
    logger.info(f'Learnable Parameters: {param}M ({ratio}%)')

    if not args.eval and not args.dump:
        world_size = comm.get_world_size()
        per_gpu_bs = int(cfg.data.train.loader.batch_size / world_size)
        cfg.data.train.loader.batch_size = per_gpu_bs
        logger.info(f'Auto Scale Batch Size: {world_size} GPU(s) * {per_gpu_bs} Samples')

    if args.dump:
        cfg.stages.validation.dump_template = 'hl_{}_submission.jsonl'
        args.eval = 'both'

    eval_results = setup_eval_capture(logger)       # 1. 注册 handler，拿到空字典

    # 2. 运行阶段
    engine = Engine(
        model,
        cfg.data,
        stages=cfg.stages,
        hooks=cfg.hooks,
        work_dir=work_dir,
        seed=seed,
        amp=args.amp,
        debug=args.debug)

    if checkpoint := args.checkpoint:
        engine.load_checkpoint(checkpoint)
    elif checkpoint := args.resume:
        engine.resume(checkpoint)

    # engine.launch(eval=args.eval)
    if args.eval:
        engine.launch(eval=args.eval_type)  # 2. 启动引擎，执行 eval 或 dump
    else:
        engine.launch()            # 启动引擎，执行训练
    
    # 3. launch 完成后，打印
    print_eval_results(eval_results, sort_by_value=False)

if __name__ == '__main__':
    main()
