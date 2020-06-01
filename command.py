#FLOWER102 command
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Flower17 -j 6 -p 64 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 50 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 1000 --var-latent-dim 64  -noise 0 --l1-weight 1 --var-gan-weight 0.01 --FID


#CIFAR 10
CUDA_VISIBLE_DEVICES=1 python main.py --dataset CIFAR10 -j 6 -p 32 -b 256 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 50 --incremental-data True --num-base-tasks 1 --num-increment-tasks 2 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 1000 --var-latent-dim 64  -noise 0 --l1-weight 1 --var-gan-weight 0.01 --FID

#debugger
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Flower17 -j 6 -p 64 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 2 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 1 --var-latent-dim 64  -noise 0 --l1-weight 1 --var-gan-weight 0.01 --FID -d --openset-generative-replay-threshold 1