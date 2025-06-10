import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# python legged_gym_/scripts/train.py --task=g1

# export LD_LIBRARY_PATH=/home/westlakeg1/anaconda3/envs/unitree-rl/lib:$LD_LIBRARY_PATH

# pip install --proxy http://10.0.1.68:8889 -e .
# pip install --proxy http://10.0.1.68:18889 -e .

# export http_proxy=http://10.0.1.68:18889
# export https_proxy=http://10.0.1.68:18889

# export LD_LIBRARY_PATH=/home/westlakeg1/anaconda3/envs/g1_gym_test/lib:$LD_LIBRARY_PATH


# python play.py --task g1_full --load_run Apr19_22-05-49_lstm11 --checkpoint 30000 --experiment_name g1_full_gru_trimesh