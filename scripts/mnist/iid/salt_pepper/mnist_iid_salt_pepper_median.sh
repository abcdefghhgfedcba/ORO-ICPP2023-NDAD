LOG_DIR='log_result'
DATA_DIR='benchmark/mnist/data'
ALG="mp_fedavg"
MODEL="cnn"
DIRTY_RATE=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
NOISE_MAGNITUDE=0.1
AGGREGATE='median'
NOISE_TYPE='salt_pepper'
MALICIOUS_CLIENT=25
ATTACKED_CLASS=(0 1 2 3 4 5 6 7 8 9)
AGG_ALGORITHM="median"
WANDB=1
ROUND=1000
EPOCH_PER_ROUND=5
BATCH=64
PROPOTION=0.2
NUM_THRESH_PER_GPU=1
NUM_GPUS=1
SERVER_GPU_ID=0
TASK="dirtymnist_iid_salt_pepper"
IDX_DIR="mnist/dirichlet/mnist_50_clients_dirichlet_03.json"

python main.py --task ${TASK} --dirty_rate ${DIRTY_RATE[@]} --noise_magnitude ${NOISE_MAGNITUDE} --model ${MODEL} --algorithm ${ALG} --wandb ${WANDB} --data_folder ${DATA_DIR} --log_folder ${LOG_DIR} --dataidx_filename ${IDX_DIR} --num_rounds ${ROUND} --num_epochs ${EPOCH_PER_ROUND} --proportion ${PROPOTION} --batch_size ${BATCH} --num_threads_per_gpu ${NUM_THRESH_PER_GPU} --num_gpus ${NUM_GPUS} --server_gpu_id ${SERVER_GPU_ID} --aggregate ${AGGREGATE} --noise_type ${NOISE_TYPE} --num_malicious ${MALICIOUS_CLIENT} --attacked_class ${ATTACKED_CLASS[@]} --agg_algorithm ${AGG_ALGORITHM} 
