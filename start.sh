cd /opt/ml/wxcode

python -u inference.py \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --ckpt_file save/model_epoch_3_mean_f1_0.6621.bin
