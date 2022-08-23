# ML-1M
CUDA_VISIBLE_DEVICES=6 nohup python main.py --dataset ml-1m --context_hops 3 --gnn ApeGNN --pool sum --Ks [20] --gpu_id 6 > ./logs/ml-1m/ApeGNN.log 2>&1 &
nohup python main.py --dataset ml-1m --gnn ApeGNN --pool sum --Ks [20] --gpu_id 4 > ./logs/ml-1m/ApeGNN_0819.log 2>&1 &

## lightgcn
CUDA_VISIBLE_DEVICES=0 nohup python main.py --dataset ml-1m --gnn lightgcn --pool sum --Ks '[20]' > ./logs/ml-1m/lightgcn_0819.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python main.py --dataset ml-1m --gnn lightgcn --pool sum --Ks '[5, 10, 20, 100]' > ./logs/ml-1m/lightgcn_0819.log 2>&1 &
## ngcf
CUDA_VISIBLE_DEVICES=0 nohup python main.py --dataset ml-1m --gnn ngcf --pool concat --Ks [20] > ./logs/ml-1m/ngcf_0819.log 2>&1 &
## ApeGNN_Per
CUDA_VISIBLE_DEVICES=1 nohup python main.py --dataset ml-1m --gnn ApeGNN_Per --pool sum --Ks [20] --step 1 --runs 1 --t_u 2 --t_i 2 > ./logs/ml-1m/ApeGNN_tu2_ti2_0819.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --dataset ml-1m --gnn ApeGNN_Per --pool sum --Ks '[20, 50]' --step 1 --runs 1 --t_u 3 --t_i 3 > ./logs/ml-1m/ApeGNN_tu3_ti3_0821.log 2>&1 &
nohup python main.py --dataset ml-1m --gnn ApeGNN_Per --pool sum --Ks '[20, 50]' --step 1 --runs 1 --t_u 2 --t_i 2 --gpu_id 2 --gs_l2 2 > ./logs/ml-1m/ApeGNN_tu2_ti2_0822.log 2>&1 &

# AMiner
nohup python main.py --dataset aminer --gnn ApeGNN_Per --pool sum --Ks [20] --step 1 --runs 1 --t_u 2 --t_i 2 > ./logs/aminer/ApeGNN_tu2_ti2_0819.log 2>&1 &
nohup python main.py --dataset aminer --gnn ApeGNN_Per --pool sum --Ks '[20, 50]' --step 1 --runs 1 --t_u 3 --t_i 3 --gpu_id 6 > ./logs/aminer/ApeGNN_tu3_ti3_0822.log 2>&1 &
nohup python main.py --dataset aminer --gnn ApeGNN_Per --pool sum --Ks '[20, 50]' --step 1 --runs 1 --t_u 2 --t_i 2 --gs_l2 1 --gpu_id 1 > ./logs/aminer/ApeGNN_tu2_ti2_0822.log 2>&1 &
nohup python main.py --dataset aminer --gnn ApeGNN_Per_Deg --pool sum --Ks '[20, 50]' --step 1 --runs 1 --gs_l2 1 --e 1e-7 --gpu_id 6 > ./logs/aminer/ApeGNN_tu2_ti2_0823.log 2>&1 &

# gowalla
nohup python main.py --dataset gowalla --gnn ApeGNN_Per --pool sum --Ks [20] --step 1 --runs 1 --t_u 2 --t_i 2 --gpu_id 6 > ./logs/gowalla/ApeGNN_tu2_ti2_0819.log 2>&1 &
nohup python main.py --dataset gowalla --gnn ApeGNN_Per --pool sum --Ks '[20, 50]' --step 1 --runs 1 --t_u 3 --t_i 3 --gpu_id 0 > ./logs/gowalla/ApeGNN_tu3_ti3_0821.log 2>&1 &
nohup python main.py --dataset gowalla --gnn ApeGNN_Per --pool sum --Ks '[20, 50]' --step 1 --runs 1 --t_u 2 --t_i 2 --gpu_id 1 --gs_l2 1 > ./logs/gowalla/ApeGNN_tu2_ti2_0822.log 2>&1 &

# yelp2018
nohup python main.py --dataset yelp2018 --gnn ApeGNN_Per --pool sum --Ks [20] --step 1 --runs 1 --t_u 2 --t_i 2 --gpu_id 5 > ./logs/yelp2018/ApeGNN_tu2_ti2_0819.log 2>&1 &
nohup python main.py --dataset yelp2018 --gnn ApeGNN_Per --pool sum --Ks '[20, 50]'  --step 1 --runs 1 --t_u 3 --t_i 3 --gpu_id 2 > ./logs/yelp2018/ApeGNN_tu3_ti3_0821.log 2>&1 &
nohup python main.py --dataset yelp2018 --gnn ApeGNN_Per --pool sum --Ks '[20, 50]' --step 1 --runs 1 --t_u 2 --t_i 2 --gpu_id 3 --gs_l2 1 > ./logs/yelp2018/ApeGNN_tu2_ti2_0822.log 2>&1 &

# amazon
nohup python main.py --dataset amazon --gnn ApeGNN_Per --pool sum --Ks [20] --step 1 --runs 1 --t_u 2 --t_i 2 --batch_size 2048 --gpu_id 4 > ./logs/amazon/ApeGNN_tu2_ti2_0819.log 2>&1 &

# ali
nohup python main.py --dataset ali --gnn ApeGNN_Per --pool sum --Ks [20] --step 1 --runs 1 --t_u 2 --t_i 2 --batch_size 2048 --gpu_id 3 > ./logs/ali/ApeGNN_tu2_ti2_0819.log 2>&1 &
