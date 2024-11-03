model_name="cardiacseg"
output_dir="/root/autodl-tmp/output/sdf-b2bn-$model_name"
data_dir="/root/autodl-tmp/ImageCHD_split_sdf"

mkdir -p $output_dir

python main_training.py \
    --arch 'vit_base' \
    --batch_size 2 \
    --dataset 'imagechd' \
    --data_dir $data_dir \
    --epoch_end 500 \
    --gpu '0' \
    --n_gpu 1 \
    --input_size 96 \
    --loss dice ce rmi \
    --lossw_dice 1.0 \
    --lossw_ce 0.5 \
    --lossw_rmi 0.1 \
    --lr 1e-4 \
    --lr_decay_epoch 450 \
    --model_name $model_name \
    --norm 'batch' \
    --num_samples 4 \
    --output_dir $output_dir \
    --val_interval 10 \
    --sigmoid_rmi \
    --rmi_epoch 450 \
    --rmi_radius 3 \
    --rmi_stride 2 \
    --plot_col 8 \
    --plot_row 4 \
    --plot_slices 3