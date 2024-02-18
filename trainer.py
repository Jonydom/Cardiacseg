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


def training(model, train_loader, val_loader, optimizer, lr_scheduler, device, args):
    loss_function = LossFunction(args)
    loss_weights = [args.lossw_dice, args.lossw_ce, args.lossw_rmi]
    metric = DiceMetric(include_background=False, reduction="mean")
    
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=args.num_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=args.num_classes)])
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    dice_metrics = []
    writer = SummaryWriter(args.output_dir)
    
    demo_dir = os.path.join(args.output_dir, 'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    for epoch in range(args.epoch_start, args.epoch_end):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epoch_end}")
        
        model.train()
        epoch_loss, epoch_dc_loss, epoch_ce_loss, epoch_rmi_loss = 0, 0, 0, 0
        
        step = 0
        for batch_data in train_loader:
            step += 1
            loss, loss_list = train_step(model, optimizer, batch_data, epoch, step, loss_function, loss_weights, demo_dir, device, args)
            epoch_loss += loss.item()
            epoch_dc_loss += loss_list[0].item()
            epoch_ce_loss += loss_list[1].item()
            epoch_rmi_loss += loss_list[2].item()
#             if args.local_rank == 0:
            print(
                    f"{step}/{len(train_loader)}"
                    f", epoch_loss: {loss.item():.4f}"
                    f", dice_loss: {loss_list[0].item():.4f}"
                    f", ce_loss: {loss_list[1].item():.4f}"
                    f", rmi_loss: {loss_list[2].item():.4f}"
                )

        lr_scheduler.step()
        epoch_loss /= step
        epoch_dc_loss /= step
        epoch_ce_loss /= step
        epoch_rmi_loss /= step
        epoch_loss_values.append([epoch_loss, epoch_dc_loss, epoch_ce_loss, epoch_rmi_loss])

        writer.add_scalar("train_loss", epoch_loss, epoch)
        writer.add_scalar("train_dc_loss", epoch_dc_loss, epoch)
        writer.add_scalar("train_ce_loss", epoch_ce_loss, epoch)
        writer.add_scalar("train_rmi_loss", epoch_rmi_loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(
                f"current epoch: {epoch + 1}, current train dice: {epoch_dc_loss:.4f}, "
                f"current train ce: {epoch_ce_loss:.4f}, current train rmi: {epoch_rmi_loss:.4f}"
            )

        if (epoch + 1) % args.val_interval == 0:
            dice_wo_bg = valid_step(model, val_loader, post_pred, post_label, metric, device, args)
            dice_metrics.append(dice_wo_bg)
            writer.add_scalar("valid_dice", dice_wo_bg, epoch)

            if dice_wo_bg > best_metric:
                best_metric = dice_wo_bg
                best_metric_epoch = epoch + 1

            checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer':        optimizer.state_dict(),
                            'scheduler':        lr_scheduler.state_dict()
                        }
            torch.save(
                        checkpoint,
                        os.path.join(
                                        args.output_dir, 
                                        f"metric_model-epoch{epoch + 1}-dice{dice_wo_bg}.pth"
                                    )
                    )
            
            print(
                    f"current valid dice without background: {dice_wo_bg:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    return


def train_step(model, optimizer, batch_data, epoch, step, loss_function, loss_weights, demo_dir, device, args):
    inputs, labels = (
        batch_data["image"].to(device),
        batch_data["label"].to(device),
    )
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss_list = loss_function(outputs, labels, epoch)
    loss = 0
    for i in range(len(loss_list)):
        loss += loss_list[i] * loss_weights[i]

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if (step - 1) % args.demo_interval == 0:
        plot_demo(inputs, outputs, labels, epoch, step, demo_dir, args)

    return loss, loss_list



def valid_step(model, val_loader, post_pred, post_label, metric, device, args):
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
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