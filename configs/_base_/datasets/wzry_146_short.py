_base_ = 'datasets'
# dataset settings
data_type = 'Grounding'
data_root = 'data/wzry_146_short/'
data = dict(
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type=data_type,
            label_path=data_root + 'wzry_146_short_train.jsonl',
            video_path=data_root + 'frames_224_1.0fps',
            cache_path=data_root + 'clip_b32_vid_k4',
            query_path=data_root + 'clip_b32_txt_k4',
            use_cache=True,
            min_video_len=5,
            fps=1.0,
            unit=1),
        # loader=dict(batch_size=128, num_workers=4, pin_memory=True, shuffle=True)),
        # loader=dict(batch_size=96, num_workers=4, pin_memory=True, shuffle=True)),
        loader=dict(batch_size=256, num_workers=4, pin_memory=True, shuffle=True)),

    val=dict(
        type=data_type,
        label_path=data_root + 'wzry_146_short_val.jsonl',
        video_path=data_root + 'frames_224_1.0fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        fps=1.0,
        unit=1,
        loader=dict(batch_size=1, num_workers=4, pin_memory=True, shuffle=False)),
    test=dict(
        type=data_type,
        label_path=data_root + 'wzry_146_short_test.jsonl',
        video_path=data_root + 'frames_224_1.0fps',
        cache_path=data_root + 'clip_b32_vid_k4',
        query_path=data_root + 'clip_b32_txt_k4',
        use_cache=True,
        fps=1.0,
        unit=1,
        loader=dict(batch_size=1, num_workers=4, pin_memory=True, shuffle=False)))
