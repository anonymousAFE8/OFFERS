python main.py --device=cuda --dataset=krnd-100 --train_dir=default --state_dict_path='krnd-100_default/SASRec.epoch=260.lr=0.001.layer=2.head=1.hidden=50.maxlen=100.pth' --tokenize_only=true --maxlen=100 --tokenizer_threshold=0.2 --tokenizer_threshold_p2=0.2
python main.py --device=cuda --dataset=krnd-100 --train_dir=default --state_dict_path='krnd-100_default/SASRec.epoch=260.lr=0.001.layer=2.head=1.hidden=50.maxlen=100.pth' --tokenize_only=true --maxlen=100 --tokenizer_threshold=0.2 --tokenizer_threshold_p2=0.1
python main.py --device=cuda --dataset=krnd-100 --train_dir=default --state_dict_path='krnd-100_default/SASRec.epoch=260.lr=0.001.layer=2.head=1.hidden=50.maxlen=100.pth' --tokenize_only=true --maxlen=100 --tokenizer_threshold=0.2 --tokenizer_threshold_p2=0.05



