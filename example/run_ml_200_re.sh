ROOT_PATH=UNIDS_PATH
DATASET=ml-1m-regen-030
DATASET_Root=ml-1m-origin
DATASET_P2=1m
max_seq_len=200
max_seq_len_double=402
tokenizer_threshold=0.3
CUDA_=1
mkdir ${ROOT_PATH}/dataset/${DATASET}/
mkdir ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/
cp ${ROOT_PATH}/dataset/${DATASET_Root}/${DATASET_P2}/sasrec_format.csv ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/

python pre_pre1_convert_4_latent_tokenizer_p1.py --root_path ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/ --dataset ${DATASET} --target_path ${ROOT_PATH}/dataset/SASRec
python pre4_tokenizer.py --dataset ${DATASET} --train_dir default --maxlen ${max_seq_len} --dropout_rate 0.2 --device cuda:1 --savename 'best_save.pth' --eval_epoch 100
python pre4_tokenizer.py --device cuda --dataset ${DATASET} --train_dir default --state_dict_path PTH --tokenize_only true --maxlen ${max_seq_len} --tokenizer_threshold ${tokenizer_threshold} --tokenizer_threshold_p2 ${tokenizer_threshold} --target_name True --output_dataset_name ${DATASET}

mkdir ${ROOT_PATH}/dataset/${DATASET}/
mkdir ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/
cp ${ROOT_PATH}/dataset/${DATASET_Root}/${DATASET_P2}/sasrec_format.csv ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/
mv ${ROOT_PATH}/${DATASET}.json ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/
cp ${ROOT_PATH}/dataset/SASRec/python/data/${DATASET}_item_id_mapping.json ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/${DATASET}_item_id_mapping.json
cp ${ROOT_PATH}/dataset/SASRec/python/data/${DATASET}_user_id_mapping.json ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/${DATASET}_user_id_mapping.json
python pre1_convert_token.py --root_path ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/
python pre2_convert_sasrec_format.py --root_path ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/
python pre3_convert_krnd.py --root_path ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/ --max_seq_len ${max_seq_len}


CUDA_VISIBLE_DEVICES=${CUDA_} python 1.Build_pretraining_dataset.py --root_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/ --max_seq_len ${max_seq_len}
CUDA_VISIBLE_DEVICES=${CUDA_} python run.py --model SASRec --dataset $DATASET
mv ${ROOT_PATH}/saved/SASRec/${DATASET}/pre-trained_embedding.ckpt ${ROOT_PATH}/dataset/${DATASET}/${DATASET_P2}/pre-trained_embedding.ckpt
CUDA_VISIBLE_DEVICES=${CUDA_} python 2.Pretrain_regenerator.py --root_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/ --K 5 --max_seq_len ${max_seq_len_double} --gpu_id ${CUDA_} --epochs 40 --batch_size 3072
CUDA_VISIBLE_DEVICES=${CUDA_} python 3.Hybrid_inference.py --root_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/ --max_seq_len ${max_seq_len_double}  --domain_count 2 --loops 5 --gpu ${CUDA_}

python fin1_convert_back_krnd.py --root_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/ --train_pth_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/train_regen.pth --output_pth sasrec_format_new.csv --reverse_mapping_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/reverse_item_mapping.pth
python fin2_convert_tokenize_final.py --root_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/ 
python fin3_replace_sasrec_format.py --root_path $ROOT_PATH/dataset/${DATASET}/${DATASET_P2}/ --origin_path $ROOT_PATH/dataset/${DATASET_Root}/${DATASET_P2}/