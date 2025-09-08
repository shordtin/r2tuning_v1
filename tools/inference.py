# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import argparse

import clip
import decord
import nncore
import torch
import numpy as np
import torchvision.transforms.functional as F
from decord import VideoReader
from nncore.engine import load_checkpoint
from nncore.nn import build_model
from tabulate import tabulate
import ffmpeg

CONFIG = 'configs/qvhighlights/r2_tuning_qvhighlights.py'
WEIGHT = 'https://huggingface.co/yeliudev/R2-Tuning/resolve/main/checkpoints/r2_tuning_qvhighlights-ed516355.pth'  # noqa

import matplotlib.pyplot as plt
def plot_scatter(data):
    """
    绘制散点图
    """
    x = list(range(len(data)))
    y = data

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y)
    plt.title('Scatter Plot of Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    # plt.show()
    plt.savefig('scatter_plot.png')  # Save the figure

def load_video(video_path, cfg):
    decord.bridge.set_bridge('torch')

    vr = VideoReader(video_path)
    stride = vr.get_avg_fps() / cfg.data.val.fps
    fm_idx = [min(round(i), len(vr) - 1) for i in np.arange(0, len(vr), stride).tolist()]
    video = vr.get_batch(fm_idx).permute(0, 3, 1, 2).float() / 255

    size = 336 if '336px' in cfg.model.arch else 224
    h, w = video.size(-2), video.size(-1)
    s = min(h, w)
    x, y = round((h - s) / 2), round((w - s) / 2)
    video = video[..., x:x + s, y:y + s]
    video = F.resize(video, size=(size, size))
    video = F.normalize(video, (0.481, 0.459, 0.408), (0.269, 0.261, 0.276))
    video = video.reshape(video.size(0), -1).unsqueeze(0)

    return video

def load_video_new(video_path, cfg):
    """
    和 _proc 对齐的新版 load_video：
    - 用 ffmpeg 实现 fps / scale / crop
    - 输出 torch.Tensor, shape = [1, N, 3*size*size], dtype=float32
    - 归一化到 ImageNet 范围，mean=(0.481,0.459,0.408), std=(0.269,0.261,0.276)
    """
    # 1. 基本参数
    size = 336 if '336px' in cfg.model.arch else 224
    target_fps = cfg.data.val.fps

    # 2. probe 原视频分辨率
    probe = ffmpeg.probe(video_path)
    info = next(s for s in probe['streams'] if s['codec_type']=='video')
    orig_w, orig_h = int(info['width']), int(info['height'])

    # 3. 先把短边缩到 size，长边按比例放大
    if orig_w > orig_h:
        new_w = int(orig_w * size / orig_h)
        new_h = size
    else:
        new_w = size
        new_h = int(orig_h * size / orig_w)

    # 4. 计算中心裁剪坐标 (长边减 size 后的偏移)
    x = (new_w - size) // 2
    y = (new_h - size) // 2

    # 5. 构建 FFmpeg 管道：fps -> scale -> crop -> rawvideo(rgb24)
    stream = (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=target_fps)
        .filter('scale', new_w, new_h)
        .crop(x, y, size, size)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    )
    out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

    # 6. 从字节流解析成 numpy array, shape=(N, size, size, 3)
    frame_count = len(out) // (size * size * 3)
    frames = (
        np
        .frombuffer(out, np.uint8)
        .reshape(frame_count, size, size, 3)
        .transpose(0, 3, 1, 2)  # -> (N, 3, size, size)
        .copy()
    )

    # 7. 转 torch tensor，归一化 + 标准化
    video = torch.from_numpy(frames).float().div_(255.0)  # [N,3,H,W]
    # ImageNet-style normalize
    mean = torch.tensor([0.481, 0.459, 0.408], device=video.device).view(1,3,1,1)
    std  = torch.tensor([0.269, 0.261, 0.276], device=video.device).view(1,3,1,1)
    video = (video - mean) / std

    # 8. 展平 H×W 并加 batch 维
    B, C, H, W = 1, *video.shape[1:]  # video.shape == (N,3,H,W)
    video = video.reshape(video.shape[0], -1).unsqueeze(0)  # [1, N, 3*H*W]

    return video

def load_video_npy(npy_path):
    """
    从 .npy 文件加载视频特征。
    原始npy形状: (N_layers, T_frames, P_patches, C_features)
    转换后输出形状: (T_frames, N_layers * P_patches * C_features)
    """
    video_features = np.load(npy_path)
    # N_layers, T_frames, P_patches, C_features = video_features.shape
    # 转换: (N, T, P, C) -> (T, N, P, C)
    video_features_t = video_features.transpose(1, 0, 2, 3)
    # 展平: (T, N, P, C) -> (T, N * P * C)
    video_features_flat = video_features_t.reshape(video_features_t.shape[0], -1)
    return torch.from_numpy(video_features_flat)

def load_query_npy(npy_path):
    """
    从 .npy 文件加载查询特征。
    原始npy形状: (N_layers, L_seq_len, C_features)
    转换后输出形状: (L_seq_len, N_layers * C_features)
    """
    query_features = np.load(npy_path)
    # N_layers, L_seq_len, C_features = query_features.shape
    # 转换: (N, L, C) -> (L, N, C)
    query_features_t = query_features.transpose(1, 0, 2)
    # 展平: (L, N, C) -> (L, N * C)
    query_features_flat = query_features_t.reshape(query_features_t.shape[0], -1)
    return torch.from_numpy(query_features_flat)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('query')
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=WEIGHT)
    args = parser.parse_args()
    return args

def main():
    """
    python tools/inference.py /data/highlight/mine/r2tuning/data/wzry_146_short/videos/0001_0498_0528.mp4 "The highlight moments of the Honor of Kings game." --config configs/wzry_146_short/r2_tuning_wzry_146_short.py --checkpoint work_dirs/r2_tuning_wzry_146_short_2/epoch_100.pth
    """
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)
    cfg.model.init = True

    if args.checkpoint.startswith('http'):
        args.checkpoint = nncore.download(args.checkpoint, out_dir='checkpoints')

    print(f'Building model from {args.config}')
    model = build_model(cfg.model, dist=False).eval()
    # model = build_model(cfg.model, dist=False)

    print(f'Loading checkpoint from {args.checkpoint}')
    model = load_checkpoint(model, args.checkpoint, warning=False)

    print(f'Loading video from {args.video}')
    # video = load_video(args.video, cfg)
    video = load_video_new(args.video, cfg)

    print(f'Query: {args.query}')
    query = clip.tokenize(args.query, truncate=True)

    device = next(model.parameters()).device
    data = dict(video=video.to(device), query=query.to(device), fps=[cfg.data.val.fps])

    with torch.inference_mode():
        pred = model(data)

    # print(pred)

    print('MR Prediction:')
    tab = [('Start time', 'End time', 'Score')]
    for b in pred['_out']['boundary'][:5].tolist():
        b[:2] = [min(max(0, n), video.size(1) / cfg.data.val.fps) for n in b[:2]]
        tab.append([round(n, 2) for n in b])
    print(tabulate(tab))

    # print('HL Prediction:')
    # print(pred['_out']['saliency'].tolist())
    # plot_scatter(pred['_out']['saliency'].tolist())

def main_npy():
    """
    python tools/inference.py /data/highlight/dataset/wzry_146_short/r2tuning_features_1d0/clip_b32_vid_k4/0001_0498_0528.npy /data/highlight/dataset/wzry_146_short/r2tuning_features_1d0/clip_b32_txt_k4/0.npy --config configs/wzry_146_short/r2_tuning_wzry_146_short.py --checkpoint work_dirs/r2_tuning_wzry_146_short_2/epoch_100.pth
    """
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)
    cfg.model.init = False

    if args.checkpoint.startswith('http'):
        args.checkpoint = nncore.download(args.checkpoint, out_dir='checkpoints')

    print(f'Building model from {args.config}')
    model = build_model(cfg.model, dist=False).eval()

    print(f'Loading checkpoint from {args.checkpoint}')
    model = load_checkpoint(model, args.checkpoint, warning=False)

    print(f'Loading video from {args.video}')
    # video = load_video(args.video, cfg)
    video = load_video_npy(args.video)
    video = video.unsqueeze(0)

    print(f'Query: {args.query}')
    # query = clip.tokenize(args.query, truncate=True)
    query = load_query_npy(args.query)
    query = query.unsqueeze(0)

    device = next(model.parameters()).device
    data = dict(video=video.to(device), query=query.to(device), fps=[cfg.data.val.fps])

    with torch.inference_mode():
        pred = model(data)

    # print(pred)

    print('MR Prediction:')
    tab = [('Start time', 'End time', 'Score')]
    for b in pred['_out']['boundary'][:5].tolist():
        b[:2] = [min(max(0, n), video.size(1) / cfg.data.val.fps) for n in b[:2]]
        tab.append([round(n, 2) for n in b])
    print(tabulate(tab))

    # print('HL Prediction:')
    # print(pred['_out']['saliency'].tolist())
    # plot_scatter(pred['_out']['saliency'].tolist())


if __name__ == '__main__':
    main()
    # main_npy()
