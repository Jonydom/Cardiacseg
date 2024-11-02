import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)
from loss import LossFunction


scaler = torch.cuda.amp.GradScaler()


def training(model, train_loader, val_loader, optimizer, lr_scheduler, args):
    loss_function = LossFunction(args)
    # loss_weights = [args.lossw_dice, args.lossw_ce, args.lossw_rmi]
    metric = DiceMetric(include_background=False, reduction="mean")
    
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=args.num_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=args.num_classes)])
    
    best_metric = -1
    best_metric_epoch = -1
    dice_metrics = []
    writer = SummaryWriter(args.output_dir)
    
    demo_dir = os.path.join(args.output_dir, 'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    for epoch in range(args.epoch_start, args.epoch_end):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epoch_end}")
        
        model.train()
        epoch_loss, loss_values = 0, []
        
        step = 0
        for batch_data in train_loader:
            step += 1
            loss, loss_list = train_step(model, optimizer, batch_data, epoch, step, loss_function, demo_dir, args)
            epoch_loss += loss.item()
            loss_values.append(loss_list)
            # epoch_dc_loss += loss_list[0].item()
            # epoch_ce_loss += loss_list[1].item()
            # epoch_rmi_loss += loss_list[2].item()
            batch_str = f"{step}/{len(train_loader)}, epoch_loss: {loss.item():.4f}"
            for idx, loss_name in enumerate(args.loss):
                batch_str += f", {loss_name}_loss: {loss_list[idx].item():.4f}"
            print(batch_str)

        lr_scheduler.step()
        epoch_loss /= step
        # avg_losses = np.mean(loss_values, axis=0)
        all_losses = torch.tensor(loss_values)
        avg_losses = torch.mean(all_losses, dim=0)

        epoch_str = f"epoch {epoch + 1} average loss: {epoch_loss:.4f}"

        writer.add_scalar("train_loss", epoch_loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
        for idx, loss_name in enumerate(args.loss):
            writer.add_scalar(f"train_{loss_name}_loss", avg_losses[idx].item(), epoch)
            epoch_str += f", current train {loss_name}: {avg_losses[idx].item():.4f}"

        print(epoch_str)

        if (epoch + 1) % args.val_interval == 0:
            dice_wo_bg = valid_step(model, val_loader, post_pred, post_label, metric, args)
            writer.add_scalar("valid_dice", dice_wo_bg, epoch)

            if dice_wo_bg > best_metric:
                best_metric = dice_wo_bg
                best_metric_epoch = epoch + 1

            checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer':        optimizer.state_dict(),
                            'scheduler':        lr_scheduler.state_dict()
                        }
            checkpoint_path = os.path.join(args.output_dir, f"metric_model-epoch{epoch + 1}-dice{dice_wo_bg}.pth")
            torch.save(checkpoint, checkpoint_path)

            dice_metrics.append((dice_wo_bg, checkpoint_path))
            dice_metrics = sorted(dice_metrics, key=lambda x: x[0], reverse=True)
            for _, path in dice_metrics[5:]:
                if os.path.isfile(path):
                    os.remove(path)
            
            print(
                    f"current valid dice without background: {dice_wo_bg:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    return


def train_step(model, optimizer, batch_data, epoch, step, loss_function, demo_dir, args):
    inputs, labels = (
        batch_data["image"].cuda(),
        batch_data["label"].cuda(),
    )
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss, loss_list = loss_function(outputs, labels, epoch)
    # loss = 0
    # for i in range(len(loss_list)):
    #     loss += loss_list[i] * loss_weights[i]

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if (step - 1) % args.demo_interval == 0:
        plot_demo(inputs, outputs, labels, epoch, step, demo_dir, args)

    return loss, loss_list



def valid_step(model, val_loader, post_pred, post_label, metric, args):
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].cuda(),
                val_data["label"].cuda(),
            )

            roi_size = args.image_size
            sw_batch_size = args.sw_batch_size
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            metric(y_pred=val_outputs, y=val_labels)
        print(metric)
        dice_wo_bg = metric.aggregate().item()
        metric.reset()

    return dice_wo_bg


def plot_demo(image, pred, label, epoch, step, demo_dir, args):
    img = image[0, 0].cpu().numpy()
    prd = pred.argmax(dim=1)[0].cpu().numpy()
    lab = label[0][0].cpu().numpy()
    col, row = args.plot_col, args.plot_row
    fig, axes = plt.subplots(ncols=col, nrows=row, figsize=(4*col, 6*row))
    for i in range(col):
        for j in range(row):
            s = args.plot_slices * (i + col * j)
            a = np.concatenate([img[:, :, s], img[:, :, s]], axis=0)
            b = np.concatenate([lab[:, :, s], prd[:, :, s]], axis=0)
            axes[j, i].imshow(a, cmap='gray')
            axes[j, i].imshow(b, cmap='hot', alpha=0.5)
            axes[j, i].set_title(f'slice = {s}')
    f = plt.gcf()
    f.savefig(f'{demo_dir}/train-epoch{epoch+1}_step{step}.jpg')
    f.clear()
    return