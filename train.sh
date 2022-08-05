rm -rf src/save/*
echo "Start pretrain..."
python src/lxmert_pretrain.py  \
    --batch_size 50 \
    --num_workers 15 \
    --learning_rate 5e-5 \
    --max_epochs 5        # 预训练
echo "Start training..."
python src/lxmert_main.py \
    --batch_size 26 \
    --num_workers 13 \
    --learning_rate 4e-5 \
    --max_epochs 5        # 微调权重
echo "Complete!"