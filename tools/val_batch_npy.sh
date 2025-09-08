# --eval：使用eval模式
# --eval_type test 使用 test npy 特征文件；val 使用 val npy 特征文件
python tools/launch_wzry_146_short.py configs/wzry_146_short/r2_tuning_wzry_146_short.py --checkpoint work_dirs/r2_tuning_wzry_146_short_2/epoch_100.pth --eval --eval_type val