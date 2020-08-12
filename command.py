#CIFAR 10
python main.py --dataset CIFAR10 -j 4 -p 32 -b 128 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 20 --incremental-data True --num-base-tasks 4 --num-increment-tasks 5 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 64 --l1-weight 1 --var-gan-weight 0.025 --var-beta 0.1 --FID --gan-loss hinge-gan --learning-rate 0.0001 --gen-learning-rate 0.0001 --dis-learning-rate 0.0001

#Flower17
# Base command
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Flower17 -j 4 -p 64 -b 32 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 20 --incremental-data True --num-base-task 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 64  --recon-weight 1 --var-gan-weight 0.1 --var-beta 0.1 --FID --gan-loss hinge-gan --learning-rate 0.0001 --gen-learning-rate 0.0001 --dis-learning-rate 0.0001 --tanh -big-wrn True
# extra-z-samples
CUDA_VISIBLE_DEVICES=0 python main.py --dataset Flower17 -j 4 -p 64 -b 32 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 20 --incremental-data True --num-base-task 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 2000 --var-latent-dim 64  --recon-weight 1 --var-gan-weight 0.1 --var-beta 0.1 --FID --gan-loss hinge-gan --learning-rate 0.0001 --gen-learning-rate 0.0001 --dis-learning-rate 0.0001 --tanh -big-wrn True --extra-z-samples



CUDA_VISIBLE_DEVICES=0 python main.py --dataset Imagewoof -j 4 -p 64 -b 32 -a WRN_gan --wrn-depth 26 --gan --visualization-epoch 20 --incremental-data True --num-base-task 1 --num-increment-tasks 1 --num-dis-feature 64 --openset-generative-replay True -genreplay True --epoch 4000 --var-latent-dim 64  --recon-weight 10 --var-gan-weight 1 --var-cls-beta 1 --var-beta 1 --FID --gan-loss hinge-gan --learning-rate 0.0001 --gen-learning-rate 0.0001 --dis-learning-rate 0.0001 --tanh -big-wrn True --blur