model_name="cardiacseg"
output_dir="output-sdf-b1in-$model_name"
data_dir="/home/jianglei/VCL-Project/data/2022Jianglei/dataset/ImageCHD_split_sdf"
pretrained_weight="pretrained_vit-b.pth"

mkdir -p $output_dir

python main_training.py \
    --arch 'vit_base' \
    --batch_size 1 \
    --dataset 'imagechd' \
    --data_dir $data_dir \
    --epoch_end 500 \
    --gpu '0,1' \
    --n_gpu 2 \
    --input_size 128 \
    --loss dice ce rmi \
    --lossw_dice 1.0 \
    --lossw_ce 0.5 \
    --lossw_rmi 0.1 \
    --lr 1e-4 \
    --lr_decay_epoch 450 \
    --model_name $model_name \
    --norm 'instance' \
    --num_samples 4 \
    --output_dir $output_dir \
    --val_interval 10 \
    --sigmoid_rmi \
    --rmi_epoch 450 \
    --rmi_radius 3 \
    --rmi_stride 2 \
    --plot_col 8 \
    --plot_row 4 \
    --plot_slices 4