#FLOWER102 command
python main.py --dataset Flower17 -j 6 -p 64 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 50 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 1000 --var-latent-dim 64  -noise 0 --l1-weight 1 --var-gan-weight 0.01 --FID
CUDA_VISIBLE_DEVICES=1 python main.py --dataset Flower17 -j 3 -p 64 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 50 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 1000 --var-latent-dim 64  --blur --l1-weight 3 --var-gan-weight 1 --FID
CUDA_VISIBLE_DEVICES=1 python main.py --dataset Flower17 -j 3 -p 64 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 5 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 64  --blur --l1-weight 1 --var-gan-weight 0.025 --dis-learning-rate 0.0001  --FID --lambda-gp 0

#CIFAR 10
CUDA_VISIBLE_DEVICES=1 python main.py --dataset CIFAR10 -j 6 -p 32 -b 256 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 50 --incremental-data True --num-base-tasks 1 --num-increment-tasks 2 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 1000 --var-latent-dim 64  -noise 0 --l1-weight 1 --var-gan-weight 0.01 --FID

#debugger
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Flower17 -j 6 -p 64 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 2 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 1 --var-latent-dim 64  -noise 0 --l1-weight 1 --var-gan-weight 0.01 --FID -d --openset-generative-replay-threshold 1




CUDA_VISIBLE_DEVICES=1 python main.py --dataset Flower17 -j 3 -p 64 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 50 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 64  --blur --FID
# DGR test
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Flower17 -j 3 -p 64 -b 32 -a WRN_enc_res_dec_gan --wrn-depth 26 --gan --visualization-epoch 20 --incremental-data True --num-base-tasks 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 64  --l1-weight 1 --var-gan-weight 1 --FID --gan-loss DGR -d


CUDA_VISIBLE_DEVICES=1 python main.py --dataset CIFAR10 -j 2 -p 32 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 20 --incremental-data True --num-base-task 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 128  --l1-weight 1 --var-gan-weight 1 --var-beta 0.1 --FID --gan-loss hinge-gan --learning-rate 0.0002 --gen-learning-rate 0.0002 --dis-learning-rate 0.0002

CUDA_VISIBLE_DEVICES=0 python main.py --dataset Flower17 -j 4 -p 64 -b 32 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 20 --incremental-data True --num-base-task 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 128  --l1-weight 1 --var-gan-weight 0.025 --var-beta 0.1 --FID --gan-loss hinge-gan --learning-rate 0.0001 --gen-learning-rate 0.0001 --dis-learning-rate 0.0001