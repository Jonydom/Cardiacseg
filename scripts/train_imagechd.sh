output_dir=/output
data_dir=/ImageCHD
pretrained_weight=/pretrained_vit-b.pth

mkdir -p $output_dir

python main_training.py \
    --arch 'vit_base' \
    --batch_size 2 \
    --dataset 'imagechd' \
    --data_dir $data_dir \
    --epoch_end 300 \
    --gpu '0,1' \
    --input_size 128 \
    --lossw_dice 1.0 \
    --lossw_ce 0.5 \
    --lossw_rmi 0.1 \
    --lr 1e-3 \
    --lr_decay_epoch 250 \
    --model_name 'cardiacseg' \
    --norm 'batch' \
    --num_samples 4 \
    --output_dir $output_dir \
    --val_interval 10 \
    --sigmoid_rmi \
    --rmi_epoch 250 \
    --rmi_radius 3 \
    --rmi_stride 2 \
    --warm_up 20 \
    --finetune \
    --pretrained_model $pretrained_weight