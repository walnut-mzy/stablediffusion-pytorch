import torch
import torch.nn as nn
from stablediffusion import stablediffusion, stable_diffusion_for_minist
from dataprocess import MNISTDataset
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
try:
    from stablediffusion import get_imagenet_model, print_model_info
    from dataprocess import ImageNetDataset
    IMAGENET_AVAILABLE = True
except ImportError:
    print("Warning: ImageNet模块未找到，仅支持MNIST训练")
    IMAGENET_AVAILABLE = False
try:
    from stablediffusion import stable_diffusion_for_cifar10
    from dataprocess import CIFAR10Dataset
    CIFAR10_AVAILABLE = True
except ImportError:
    print("Warning: CIFAR-10模块未找到，仅支持MNIST和ImageNet训练")
    CIFAR10_AVAILABLE = False

def print_training_parameters(config):
    print("\n" + "="*60)
    dataset_name = config.get('dataset_name', 'MNIST')
    print(f"{dataset_name} Stable Diffusion 两阶段训练配置")
    print("="*60)
    
    print(f"\n[基础参数]")
    print(f"设备: {config['device']}")
    print(f"批大小: {config['batch_size']}")
    print(f"总训练轮次: {config['epochs']}")
    print(f"VAE训练轮次: {config.get('vae_epochs', 10)}")
    print(f"扩散模型训练轮次: {config.get('diffusion_epochs', config['epochs'] - config.get('vae_epochs', 10))}")
    print(f"保存模型路径: {config['save_dir']}")
    print(f"训练模式: 两阶段训练 (VAE → 扩散模型)")
    
    print(f"\n[优化器参数]")
    print(f"VAE优化器类型: AdamW")
    print(f"VAE学习率: 1e-3")
    print(f"扩散模型优化器类型: {config['optimizer_type']}")
    print(f"扩散模型学习率: {config['learning_rate']}")
    print(f"权重衰减: {config['weight_decay']}")
    print(f"使用学习率调度器: {config['use_scheduler']}")
    if config['use_scheduler']:
        print(f"调度器类型: {config['scheduler_type']}")
    
    if config.get('use_classifier_guidance', False):
        print(f"\n[Classifier Guidance参数]")
        print(f"使用Classifier Guidance: {config['use_classifier_guidance']}")
        print(f"分类器学习率: {config.get('classifier_lr', 'N/A')}")
        print(f"分类器权重衰减: {config.get('classifier_weight_decay', 'N/A')}")
        print(f"Guidance强度: {config.get('guidance_scale', 'N/A')}")
    
    print(f"\n[模型参数]")
    print(f"模型名称: {config.get('model_name', 'MNIST_SD')}")
    print(f"总参数量: {config['total_params']:,}")
    print(f"输入图像尺寸: {config['input_image_size']}x{config['input_image_size']}")
    print(f"潜在空间尺寸: {config['latent_size']}x{config['latent_size']}")
    print(f"潜在空间通道数: {config['latent_channels']}")
    print(f"扩散步数: {config['num_timesteps']}")
    print(f"条件生成: {'启用' if config['use_conditional_generation'] else '禁用'}")
    if config['use_conditional_generation']:
        print(f"类别数量: {config['num_classes']}")
    
    print(f"\n[数据集参数]")
    print(f"数据集: {dataset_name}")
    print(f"训练集大小: {config['train_size']}")
    print(f"验证集大小: {config['test_size']}")
    print(f"数据使用比例: {config.get('data_percentage', 100.0):.1f}%")
    print(f"数据增强: {config['use_augmentation']}")
    print(f"归一化范围: {config['normalization']}")
    if config.get('subset_classes') is not None:
        print(f"使用类别子集: {len(config['subset_classes'])} 个类别")
    print("="*60 + "\n")
    
def train_one_epoch(model, dataloader, optimizer, device, epoch, writer=None, training_mode='diffusion', 
                   classifier_optimizer=None, classifier_weight=1.0, gradient_clip=1.0):
    model.train()
    total_loss = 0
    total_diffusion_loss = 0
    total_classifier_loss = 0
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if labels is not None else None
        
        total_batch_loss = 0
        diffusion_loss = 0
        classifier_loss = 0
        
        try:
            if training_mode == 'diffusion':
                loss = model(images, mode='diffusion', labels=labels)
                diffusion_loss = loss.item()
                total_batch_loss = loss
                
            elif training_mode == 'reconstruction':
                reconstructed = model(images, mode='reconstruction')
                loss = F.mse_loss(reconstructed, images)
                total_batch_loss = loss
                
            elif training_mode == 'classifier' and model.classifier is not None:
                loss = model(images, mode='classifier', labels=labels)
                classifier_loss = loss.item()
                total_batch_loss = loss
                
            elif training_mode == 'joint' and model.classifier is not None:
                diff_loss = model(images, mode='diffusion', labels=labels)
                diffusion_loss = diff_loss.item()
                
                class_loss = model(images, mode='classifier', labels=labels)
                classifier_loss = class_loss.item()
                
                total_batch_loss = diff_loss + classifier_weight * class_loss
            else:
                raise ValueError(f"Invalid training mode: {training_mode}")
            
            # 梯度清零和反向传播
            if training_mode == 'classifier' and classifier_optimizer is not None:
                classifier_optimizer.zero_grad()
            elif training_mode == 'joint' and classifier_optimizer is not None:
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            total_batch_loss.backward()
            
            # 梯度裁剪
            if gradient_clip > 0:
                if training_mode == 'classifier' and classifier_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), gradient_clip)
                elif training_mode == 'joint':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for name, p in model.named_parameters() if 'classifier' not in name], 
                        gradient_clip
                    )
            
            # 优化器步进
            if training_mode == 'classifier' and classifier_optimizer is not None:
                classifier_optimizer.step()
            elif training_mode == 'joint' and classifier_optimizer is not None:
                optimizer.step()
                classifier_optimizer.step()
            else:
                optimizer.step()
            
            # 更新EMA
            if training_mode in ['diffusion', 'joint']:
                model.diffusion_model.update_ema()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"显存不足，跳过batch {batch_idx}")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        total_loss += total_batch_loss.item()
        total_diffusion_loss += diffusion_loss
        total_classifier_loss += classifier_loss
        
        postfix_dict = {"total_loss": f"{total_batch_loss.item():.4f}"}
        if diffusion_loss > 0:
            postfix_dict["diff_loss"] = f"{diffusion_loss:.4f}"
        if classifier_loss > 0:
            postfix_dict["class_loss"] = f"{classifier_loss:.4f}"
        pbar.set_postfix(postfix_dict)
        
        # 记录到TensorBoard
        if writer is not None and batch_idx % 50 == 0:  # 降低记录频率
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/Train_Batch_Total', total_batch_loss.item(), global_step)
            if diffusion_loss > 0:
                writer.add_scalar('Loss/Train_Batch_Diffusion', diffusion_loss, global_step)
            if classifier_loss > 0:
                writer.add_scalar('Loss/Train_Batch_Classifier', classifier_loss, global_step)
    
    avg_loss = total_loss / len(dataloader)
    avg_diffusion_loss = total_diffusion_loss / len(dataloader)
    avg_classifier_loss = total_classifier_loss / len(dataloader)
    elapsed = time.time() - start_time
    
    if writer is not None:
        writer.add_scalar('Loss/Train_Epoch_Total', avg_loss, epoch)
        if avg_diffusion_loss > 0:
            writer.add_scalar('Loss/Train_Epoch_Diffusion', avg_diffusion_loss, epoch)
        if avg_classifier_loss > 0:
            writer.add_scalar('Loss/Train_Epoch_Classifier', avg_classifier_loss, epoch)
        writer.add_scalar('Time/Epoch_Duration', elapsed, epoch)
    
    print(f"Epoch {epoch+1} - 总损失: {avg_loss:.4f}, 用时: {elapsed:.2f}秒")
    if avg_diffusion_loss > 0:
        print(f"  扩散损失: {avg_diffusion_loss:.4f}")
    if avg_classifier_loss > 0:
        print(f"  分类器损失: {avg_classifier_loss:.4f}")
    
    return avg_loss

def evaluate(model, dataloader, device, epoch=None, writer=None, training_mode='diffusion', 
             classifier_weight=1.0, max_batches=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluation"):
            if max_batches is not None and batch_count >= max_batches:
                break
                
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) if labels is not None else None
            
            try:
                if training_mode == 'classifier' and model.classifier is not None:
                    loss = model(images, mode='classifier', labels=labels)
                    
                    if labels is not None:
                        latent_shape = model.get_encoder_output_shape(images.shape)
                        noise = torch.randn(latent_shape, device=images.device)
                        latent = model.encoder(images, noise)
                        logits = model.classifier(latent)
                        pred = logits.argmax(dim=1)
                        total_correct += (pred == labels).sum().item()
                        total_samples += labels.size(0)
                        
                elif training_mode == 'joint' and model.classifier is not None:
                    diff_loss = model(images, mode='diffusion', labels=labels)
                    class_loss = model(images, mode='classifier', labels=labels)
                    loss = diff_loss + classifier_weight * class_loss
                    
                    if labels is not None:
                        latent_shape = model.get_encoder_output_shape(images.shape)
                        noise = torch.randn(latent_shape, device=images.device)
                        latent = model.encoder(images, noise)
                        logits = model.classifier(latent)
                        pred = logits.argmax(dim=1)
                        total_correct += (pred == labels).sum().item()
                        total_samples += labels.size(0)
                        
                else:
                    if training_mode == 'reconstruction':
                        reconstructed = model(images, mode='reconstruction')
                        loss = F.mse_loss(reconstructed, images)
                    else:
                        loss = model(images, mode=training_mode, labels=labels)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"评估时显存不足，跳过batch {batch_count}")
                    torch.cuda.empty_cache()
                    batch_count += 1
                    continue
                else:
                    raise e
            
            total_loss += loss.item()
            batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
        if total_samples > 0:
            accuracy = total_correct / total_samples
            writer.add_scalar('Accuracy/Validation', accuracy, epoch)
            print(f"评估 - 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        else:
            print(f"评估 - 平均损失: {avg_loss:.4f}")
    else:
        print(f"评估 - 平均损失: {avg_loss:.4f}")
    
    return avg_loss

def save_model(model, optimizer, epoch, loss, save_path, classifier_optimizer=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if classifier_optimizer is not None:
        checkpoint['classifier_optimizer_state_dict'] = classifier_optimizer.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"模型已保存到: {save_path}")

def generate_and_log_samples(model, dataset, device, epoch, writer, save_dir, 
                           batch_size=5, channels=3, image_size=32):
    print(f"生成第{epoch+1}轮的样本图像...")
    model.eval()
    
    with torch.no_grad():
        # 如果模型支持条件生成，生成随机类别标签
        y = None
        if hasattr(model, 'num_classes') and model.num_classes is not None:
            y = torch.randint(0, model.num_classes, (batch_size,), device=device)
        
        latent_samples = model.sample_latent(
            batch_size=batch_size,
            device=device,
            y=y
        )
        
        final_latent = latent_samples[-1]
        # 确保tensor在正确的设备上
        if not final_latent.is_cuda and device.type == 'cuda':
            final_latent = final_latent.to(device)
        
        generated_images = model.decoder(final_latent)
    
    denorm_images = dataset.denormalize(generated_images)
    
    grid = vutils.make_grid(denorm_images, nrow=5, padding=2, normalize=False)
    
    writer.add_image(f'Generated_Images/Epoch_{epoch+1}', grid, epoch)
    
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 3))
    plt.axis("off")
    plt.title(f"generation samples - Epoch {epoch+1}")
    
    if channels == 3:
        img_grid = grid.cpu().permute(1, 2, 0).numpy()
        img_grid = np.clip(img_grid, 0, 1)
        plt.imshow(img_grid)
    else:
        plt.imshow(grid.cpu().squeeze().numpy(), cmap='gray')
    
    sample_path = os.path.join(sample_dir, f"epoch_{epoch+1}_samples.png")
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"样本图像已保存到: {sample_path}")
    
    for i in range(batch_size):
        single_img_path = os.path.join(sample_dir, f"epoch_{epoch+1}_sample_{i+1}.png")
        plt.figure(figsize=(3, 3))
        plt.axis("off")
        
        if channels == 3:
            img = denorm_images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
        else:
            plt.imshow(denorm_images[i].cpu().squeeze().numpy(), cmap='gray')
        
        plt.savefig(single_img_path, dpi=150, bbox_inches='tight')
        plt.close()

def generate_classifier_guided_samples(model, dataset, device, epoch, writer, save_dir, 
                                      guidance_scales=[0.0, 1.0, 5.0, 10.0], samples_per_class=2, 
                                      channels=3, image_size=32):
    if model.classifier is None:
        print("跳过classifier guidance生成 - 分类器未启用")
        return
    
    print(f"生成第{epoch+1}轮的Classifier Guided样本...")
    model.eval()
    
    num_classes = model.num_classes
    
    for guidance_scale in guidance_scales:
        print(f"  生成guidance_scale={guidance_scale}的样本...")
        
        all_samples = []
        all_labels = []
        
        with torch.no_grad():
            for class_idx in range(num_classes):
                if guidance_scale > 0:
                    samples = model.sample_with_classifier_guidance(
                        batch_size=samples_per_class,
                        device=device,
                        target_class=class_idx,
                        guidance_scale=guidance_scale,
                        use_ema=True
                    )
                else:
                    y = torch.full((samples_per_class,), class_idx, device=device, dtype=torch.long)
                    samples = model.sample_latent(
                        batch_size=samples_per_class,
                        device=device,
                        y=y,
                        use_ema=True
                    )
                
                final_latent = samples[-1]
                # 确保tensor在正确的设备上
                if not final_latent.is_cuda and device.type == 'cuda':
                    final_latent = final_latent.to(device)
                generated_images = model.decoder(final_latent)
                all_samples.append(generated_images)
                all_labels.extend([class_idx] * samples_per_class)
        
        all_samples = torch.cat(all_samples, dim=0)
        
        denorm_images = dataset.denormalize(all_samples)
        
        grid = vutils.make_grid(denorm_images, nrow=num_classes, padding=2, normalize=False)
        
        writer.add_image(f'Classifier_Guided_Images/Epoch_{epoch+1}_Scale_{guidance_scale}', grid, epoch)
        
        guided_dir = os.path.join(save_dir, "classifier_guided_samples")
        os.makedirs(guided_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 6))
        plt.axis("off")
        plt.title(f"Classifier Guided samples - Epoch {epoch+1}, Guidance Scale: {guidance_scale}\nnumbers 0-9")
        
        if channels == 3:
            img_grid = grid.cpu().permute(1, 2, 0).numpy()
            img_grid = np.clip(img_grid, 0, 1)
            plt.imshow(img_grid)
        else:
            plt.imshow(grid.cpu().squeeze().numpy(), cmap='gray')
        
        guided_path = os.path.join(guided_dir, f"epoch_{epoch+1}_guidance_{guidance_scale}.png")
        plt.savefig(guided_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Guidance scale {guidance_scale}样本已保存到: {guided_path}")

def generate_reconstruction_samples(model, dataset, device, epoch, writer, save_dir, 
                                  batch_size=5, channels=3, image_size=32):
    print(f"生成第{epoch+1}轮的重建样本...")
    model.eval()
    
    dataloader = dataset.get_test_loader()
    real_images, _ = next(iter(dataloader))
    real_images = real_images[:batch_size].to(device)
    
    with torch.no_grad():
        reconstructed_images = model(real_images, mode='reconstruction')
    
    real_denorm = dataset.denormalize(real_images)
    recon_denorm = dataset.denormalize(reconstructed_images)
    
    comparison = torch.stack([real_denorm, recon_denorm], dim=1).view(-1, *real_denorm.shape[1:])
    
    grid = vutils.make_grid(comparison, nrow=10, padding=2, normalize=False)
    
    writer.add_image(f'Reconstruction/Epoch_{epoch+1}', grid, epoch)
    
    recon_dir = os.path.join(save_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    plt.axis("off")
    plt.title(f"compare - Epoch {epoch+1}\n(orgin-construction-orgin-construction...)")
    
    if channels == 3:
        img_grid = grid.cpu().permute(1, 2, 0).numpy()
        img_grid = np.clip(img_grid, 0, 1)
        plt.imshow(img_grid)
    else:
        plt.imshow(grid.cpu().squeeze().numpy(), cmap='gray')
    
    recon_path = os.path.join(recon_dir, f"epoch_{epoch+1}_reconstruction.png")
    plt.savefig(recon_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Reconstruction samples saved to: {recon_path}")

def train_two_stage_model(model, dataset, optimizer, scheduler, device, epochs, save_dir, writer,
                         vae_epochs=10, diffusion_epochs=None, classifier_optimizer=None, 
                         classifier_scheduler=None, classifier_weight=1.0, guidance_scales=[0.0, 1.0, 5.0], 
                         gradient_clip=1.0, dataset_name='MNIST'):
    """
    Two-stage training: first train VAE, then freeze VAE and train diffusion model
    
    Args:
        model: The stable diffusion model
        dataset: Dataset loader
        optimizer: Main optimizer (for diffusion model in stage 2)
        scheduler: Learning rate scheduler
        device: Training device
        epochs: Total epochs
        save_dir: Directory to save models
        writer: Tensorboard writer
        vae_epochs: Number of epochs to train VAE (stage 1)
        diffusion_epochs: Number of epochs to train diffusion (stage 2), if None uses remaining epochs
        ...
    """
    if diffusion_epochs is None:
        diffusion_epochs = epochs - vae_epochs
    
    print(f"\n{'='*60}")
    print("开始两阶段训练:")
    print(f"阶段1: VAE训练 ({vae_epochs} epochs)")
    print(f"阶段2: 扩散模型训练 ({diffusion_epochs} epochs)")
    print('='*60)
    
    # =================== 阶段1: 训练VAE ===================
    print(f"\n{'='*50}")
    print("阶段1: 训练VAE重建模型")
    print('='*50)
    
    # 创建VAE优化器
    vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    vae_optimizer = torch.optim.AdamW(vae_params, lr=1e-3, weight_decay=1e-4)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optimizer, T_max=vae_epochs, eta_min=1e-6)
    
    best_vae_loss = float('inf')
    
    for epoch in range(vae_epochs):
        print(f"\n{'='*30}")
        print(f"VAE训练 Epoch {epoch+1}/{vae_epochs}")
        print('='*30)
        
        # VAE训练
        vae_loss = train_one_epoch(
            model=model,
            dataloader=dataset.get_train_loader(),
            optimizer=vae_optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            training_mode='reconstruction',
            gradient_clip=gradient_clip
        )
        
        # VAE验证
        vae_eval_loss = evaluate(
            model=model,
            dataloader=dataset.get_test_loader(),
            device=device,
            epoch=epoch,
            writer=writer,
            training_mode='reconstruction',
            max_batches=50 if dataset_name == 'ImageNet' else None
        )
        
        if vae_scheduler is not None:
            vae_scheduler.step()
            current_lr = vae_optimizer.param_groups[0]['lr']
            writer.add_scalar('VAE_Learning_Rate', current_lr, epoch)
        
        # 生成重建样本
        if (epoch + 1) % max(1, vae_epochs // 5) == 0:  # 每20%的训练进度生成一次
            try:
                generate_reconstruction_samples(
                    model=model,
                    dataset=dataset,
                    device=device,
                    epoch=epoch,
                    writer=writer,
                    save_dir=save_dir,
                    batch_size=5,
                    channels=model.out_channels,
                    image_size=model.input_image_size
                )
            except Exception as e:
                print(f"VAE重建样本生成失败 (epoch {epoch+1}): {e}")
        
        # 保存最佳VAE模型
        if vae_eval_loss < best_vae_loss:
            best_vae_loss = vae_eval_loss
            vae_model_path = os.path.join(save_dir, "best_vae_model.pt")
            save_model(
                model=model,
                optimizer=vae_optimizer,
                epoch=epoch,
                loss=vae_eval_loss,
                save_path=vae_model_path
            )
            print(f"新的最佳VAE模型已保存! 损失: {best_vae_loss:.4f}")
        
        print(f"VAE训练损失: {vae_loss:.4f} | VAE验证损失: {vae_eval_loss:.4f}")
    
    # 保存VAE阶段完成的模型
    vae_final_path = os.path.join(save_dir, "vae_stage_complete.pt")
    save_model(
        model=model,
        optimizer=vae_optimizer,
        epoch=vae_epochs-1,
        loss=best_vae_loss,
        save_path=vae_final_path
    )
    
    # 保存VAE权重文件（单独保存，方便后续使用）
    vae_weights_path = os.path.join(save_dir, "vae_weights.pt")
    model.save_vae_weights(vae_weights_path)
    
    print(f"VAE阶段完成，模型保存到: {vae_final_path}")
    print(f"VAE权重保存到: {vae_weights_path}")
    
    # =================== 阶段2: 冻结VAE，训练扩散模型 ===================
    print(f"\n{'='*50}")
    print("阶段2: 冻结VAE，训练扩散模型")
    print('='*60)
    
    # 冻结VAE参数
    model.freeze_vae_parameters()
    
    print("VAE参数已冻结")
    
    # 重新创建优化器，只优化扩散模型参数
    diffusion_params = [p for name, p in model.named_parameters() 
                       if p.requires_grad and 'classifier' not in name]
    diffusion_optimizer = torch.optim.AdamW(diffusion_params, lr=optimizer.param_groups[0]['lr'], 
                                          weight_decay=1e-4)
    diffusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        diffusion_optimizer, T_max=diffusion_epochs, eta_min=1e-6)
    
    print(f"扩散模型可训练参数数量: {sum(p.numel() for p in diffusion_params):,}")
    
    best_diffusion_loss = float('inf')
    
    for epoch in range(diffusion_epochs):
        actual_epoch = vae_epochs + epoch
        print(f"\n{'='*30}")
        print(f"扩散模型训练 Epoch {epoch+1}/{diffusion_epochs} (总Epoch {actual_epoch+1})")
        print('='*30)
        
        # 扩散模型训练
        diffusion_loss = train_one_epoch(
            model=model,
            dataloader=dataset.get_train_loader(),
            optimizer=diffusion_optimizer,
            device=device,
            epoch=actual_epoch,
            writer=writer,
            training_mode='diffusion',
            classifier_optimizer=classifier_optimizer,
            classifier_weight=classifier_weight,
            gradient_clip=gradient_clip
        )
        
        # 扩散模型验证
        diffusion_eval_loss = evaluate(
            model=model,
            dataloader=dataset.get_test_loader(),
            device=device,
            epoch=actual_epoch,
            writer=writer,
            training_mode='diffusion',
            classifier_weight=classifier_weight,
            max_batches=50 if dataset_name == 'ImageNet' else None
        )
        
        if diffusion_scheduler is not None:
            diffusion_scheduler.step()
            current_lr = diffusion_optimizer.param_groups[0]['lr']
            writer.add_scalar('Diffusion_Learning_Rate', current_lr, actual_epoch)
            
        if classifier_scheduler is not None:
            classifier_scheduler.step()
            classifier_lr = classifier_optimizer.param_groups[0]['lr']
            writer.add_scalar('Classifier_Learning_Rate', classifier_lr, actual_epoch)
        
        # 生成扩散样本
        if (epoch + 1) % max(1, diffusion_epochs // 10) == 0:  # 每10%的训练进度生成一次
            try:
                generate_and_log_samples(
                    model=model,
                    dataset=dataset,
                    device=device,
                    epoch=actual_epoch,
                    writer=writer,
                    save_dir=save_dir,
                    batch_size=5,
                    channels=model.out_channels,
                    image_size=model.input_image_size
                )
                
                # 如果有分类器，也生成分类器引导样本
                if model.classifier is not None and (epoch + 1) % max(1, diffusion_epochs // 5) == 0:
                    generate_classifier_guided_samples(
                        model=model,
                        dataset=dataset,
                        device=device,
                        epoch=actual_epoch,
                        writer=writer,
                        save_dir=save_dir,
                        guidance_scales=guidance_scales,
                        samples_per_class=2,
                        channels=model.out_channels,
                        image_size=model.input_image_size
                    )
                    
            except Exception as e:
                print(f"扩散样本生成失败 (epoch {actual_epoch+1}): {e}")
        
        # 保存最佳扩散模型
        if diffusion_eval_loss < best_diffusion_loss:
            best_diffusion_loss = diffusion_eval_loss
            diffusion_model_path = os.path.join(save_dir, "best_diffusion_model.pt")
            save_model(
                model=model,
                optimizer=diffusion_optimizer,
                epoch=actual_epoch,
                loss=diffusion_eval_loss,
                save_path=diffusion_model_path,
                classifier_optimizer=classifier_optimizer
            )
            print(f"新的最佳扩散模型已保存! 损失: {best_diffusion_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"diffusion_checkpoint_epoch_{actual_epoch+1}.pt")
            save_model(
                model=model,
                optimizer=diffusion_optimizer,
                epoch=actual_epoch,
                loss=diffusion_eval_loss,
                save_path=checkpoint_path,
                classifier_optimizer=classifier_optimizer
            )
        
        print(f"扩散训练损失: {diffusion_loss:.4f} | 扩散验证损失: {diffusion_eval_loss:.4f}")
    
    # 保存最终完整模型
    final_model_path = os.path.join(save_dir, "final_complete_model.pt")
    save_model(
        model=model,
        optimizer=diffusion_optimizer,
        epoch=vae_epochs + diffusion_epochs - 1,
        loss=best_diffusion_loss,
        save_path=final_model_path,
        classifier_optimizer=classifier_optimizer
    )
    
    print(f"\n两阶段训练完成!")
    print(f"最佳VAE损失: {best_vae_loss:.4f}")
    print(f"最佳扩散模型损失: {best_diffusion_loss:.4f}")
    print(f"完整模型保存到: {final_model_path}")

def main_train(training_mode, use_classifier_guidance):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 16
    epochs = 50
    vae_epochs = 10  # VAE训练轮次
    diffusion_epochs = epochs - vae_epochs  # 扩散模型训练轮次
    save_dir = "/root/tf-logs"
    guidance_scales = [0.0, 1.0, 3.0, 5.0]
    num_classes=10
    model = stable_diffusion_for_minist(use_classifier_guidance=use_classifier_guidance)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Stable Diffusion模型参数数量: {total_params:,}")
    print(f"使用Classifier Guidance: {use_classifier_guidance}")
    print(f"设备: {device}")
    
    latent_shape = model.get_encoder_output_shape((1, 3, 32, 32))
    latent_size = latent_shape[2]
    
    dataset = MNISTDataset(
        root_dir="./data",
        image_size=model.input_image_size,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        use_augmentation=True,
        convert_to_rgb=True,
        data_percentage=100.0,
    )
    
    optimizer_type = "AdamW"
    learning_rate = 1e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if 'classifier' not in name],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    classifier_optimizer = None
    classifier_lr = 1e-3
    classifier_weight_decay = 1e-4
    classifier_weight = 1.0
    
    if use_classifier_guidance and model.classifier is not None:
        classifier_optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=classifier_lr,
            weight_decay=classifier_weight_decay
        )
    
    use_scheduler = True
    scheduler_type = "CosineAnnealingLR"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=diffusion_epochs, eta_min=1e-6  # 只为扩散模型设置调度器
    )
    
    classifier_scheduler = None
    if classifier_optimizer is not None:
        classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=diffusion_epochs, eta_min=1e-6
        )
    
    dataset_info = dataset.get_dataset_info()
    
    training_config = {
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
        "vae_epochs": vae_epochs,
        "diffusion_epochs": diffusion_epochs,
        "save_dir": save_dir,
        "dataset_name": "MNIST",
        
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_scheduler": use_scheduler,
        "scheduler_type": scheduler_type if use_scheduler else None,
        
        "use_classifier_guidance": use_classifier_guidance,
        "classifier_lr": classifier_lr if use_classifier_guidance else None,
        "classifier_weight_decay": classifier_weight_decay if use_classifier_guidance else None,
        "guidance_scale": guidance_scales,
        "classifier_weight": classifier_weight,
        
        "total_params": total_params,
        "input_image_size": model.input_image_size,
        "latent_channels": model.latent_channels,
        "latent_size": latent_size,
        "num_timesteps": model.diffusion_model.num_timesteps,
        "use_conditional_generation": True,
        "num_classes": num_classes,
        
        "train_size": dataset_info["train_size"],
        "test_size": dataset_info["test_size"],
        "use_augmentation": dataset.use_augmentation,
        "normalization": dataset_info["normalization"]
    }
    
    print_training_parameters(training_config)

    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = os.path.join(save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard日志保存到: {log_dir}")
    print("可以使用以下命令启动TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    
    config_text = "\n".join([f"{k}: {v}" for k, v in training_config.items()])
    writer.add_text("Training_Config", config_text, 0)
    
    print(f"\n开始两阶段训练...\n")
    
    train_two_stage_model(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        save_dir=save_dir,
        writer=writer,
        vae_epochs=vae_epochs,
        diffusion_epochs=diffusion_epochs,
        classifier_optimizer=classifier_optimizer,
        classifier_scheduler=classifier_scheduler,
        classifier_weight=classifier_weight,
        guidance_scales=guidance_scales,
        gradient_clip=1.0,
        dataset_name='MNIST'
    )
    
    print("\n训练完成! 生成最终样本...")
    
    generate_and_log_samples(
        model=model,
        dataset=dataset,
        device=device,
        epoch=epochs-1,
        writer=writer,
        save_dir=save_dir,
        batch_size=10,
        channels=model.out_channels,
        image_size=model.input_image_size
    )
    
    generate_reconstruction_samples(
        model=model,
        dataset=dataset,
        device=device,
        epoch=epochs-1,
        writer=writer,
        save_dir=save_dir,
        batch_size=10,
        channels=model.out_channels,
        image_size=model.input_image_size
    )
    
    if use_classifier_guidance:
        generate_classifier_guided_samples(
            model=model,
            dataset=dataset,
            device=device,
            epoch=epochs-1,
            writer=writer,
            save_dir=save_dir,
            guidance_scales=guidance_scales,
            samples_per_class=3,
            channels=model.out_channels,
            image_size=model.input_image_size
        )
    
    writer.close()
    
    print(f"模型保存在: {save_dir}")
    print(f"TensorBoard日志: {log_dir}")

def main_train_imagenet(
    image_size=64,
    num_classes=1000,
    subset_classes=None,
    data_percentage=100.0,
    training_mode='two_stage',
    use_classifier_guidance=False,
    model_size='normal'
):
    """ImageNet训练主函数"""
    
    if not IMAGENET_AVAILABLE:
        print("错误: ImageNet模块未找到，无法进行ImageNet训练")
        return
    
    # 设备和基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 训练配置 - 根据图像大小调整VAE训练轮次
    if image_size <= 64:
        batch_size = 32
        epochs = 100
        vae_epochs = 15  # 更多VAE训练轮次
    elif image_size <= 128:
        batch_size = 16
        epochs = 150
        vae_epochs = 20
    else:
        batch_size = 8
        epochs = 200
        vae_epochs = 25
    
    diffusion_epochs = epochs - vae_epochs
    save_dir = f"/root/tf-logs/imagenet_{image_size}x{image_size}"
    guidance_scales = [0.0, 1.0, 3.0, 5.0]
    
    # 创建模型
    print(f"创建模型: {image_size}x{image_size}, {num_classes} classes, {model_size} size")
    model, model_name = get_imagenet_model(
        image_size=image_size,
        num_classes=num_classes,
        use_classifier_guidance=use_classifier_guidance,
        model_size=model_size
    )
    model = model.to(device)
    
    # 打印模型信息
    print_model_info(model, model_name)
    
    # 创建数据集
    dataset = ImageNetDataset(
        root_dir="./data/imagenet",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        use_augmentation=True,
        normalize_to_neg_one_to_one=True,
        data_percentage=data_percentage,
        subset_classes=subset_classes,
        center_crop=True
    )
    
    # 优化器配置
    learning_rate = 2e-4 if image_size <= 64 else 1e-4
    weight_decay = 1e-4
    
    optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if 'classifier' not in name],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    classifier_optimizer = None
    classifier_lr = 1e-3
    classifier_weight = 1.0
    
    if use_classifier_guidance and model.classifier is not None:
        classifier_optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=classifier_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    # 学习率调度器 - 只为扩散阶段设置
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=diffusion_epochs, eta_min=1e-6
    )
    
    classifier_scheduler = None
    if classifier_optimizer is not None:
        classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=diffusion_epochs, eta_min=1e-6
        )
    
    # 获取数据集信息
    dataset_info = dataset.get_dataset_info()
    model_info = {
        'total_params': sum(p.numel() for p in model.parameters()),
        'input_image_size': model.input_image_size,
        'latent_channels': model.latent_channels,
        'num_classes': model.num_classes,
        'use_conditional_generation': model.num_classes is not None,
        'num_timesteps': model.diffusion_model.num_timesteps,
    }
    
    # 计算潜在空间大小
    latent_shape = model.get_encoder_output_shape((1, 3, image_size, image_size))
    latent_size = latent_shape[2]
    
    # 训练配置
    training_config = {
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
        "vae_epochs": vae_epochs,
        "diffusion_epochs": diffusion_epochs,
        "save_dir": save_dir,
        "model_name": model_name,
        "dataset_name": "ImageNet",
        
        "optimizer_type": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_scheduler": True,
        "scheduler_type": "CosineAnnealingLR",
        
        "use_classifier_guidance": use_classifier_guidance,
        "classifier_lr": classifier_lr if use_classifier_guidance else None,
        "classifier_weight_decay": weight_decay if use_classifier_guidance else None,
        "guidance_scale": guidance_scales,
        "classifier_weight": classifier_weight,
        
        **model_info,
        "latent_size": latent_size,
        
        **dataset_info
    }
    
    # 打印训练配置
    print_training_parameters(training_config)
    
    # 创建保存目录和TensorBoard
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir)
    
    print(f"\nTensorBoard日志保存到: {log_dir}")
    print("可以使用以下命令启动TensorBoard:")
    print(f"tensorboard --logdir={log_dir}")
    
    # 记录配置到TensorBoard
    config_text = "\n".join([f"{k}: {v}" for k, v in training_config.items()])
    writer.add_text("Training_Config", config_text, 0)
    
    print(f"\n开始ImageNet两阶段训练...\n")
    
    # 开始训练
    train_two_stage_model(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        save_dir=save_dir,
        writer=writer,
        vae_epochs=vae_epochs,
        diffusion_epochs=diffusion_epochs,
        classifier_optimizer=classifier_optimizer,
        classifier_scheduler=classifier_scheduler,
        classifier_weight=classifier_weight,
        guidance_scales=guidance_scales,
        gradient_clip=1.0,
        dataset_name='ImageNet'
    )
    
    print("\n训练完成!")
    
    # 生成最终样本
    print("生成最终样本...")
    try:
        generate_and_log_samples(
            model=model,
            dataset=dataset,
            device=device,
            epoch=epochs-1,
            writer=writer,
            save_dir=save_dir,
            batch_size=8,
            channels=model.out_channels,
            image_size=model.input_image_size
        )
        
        if use_classifier_guidance:
            # 为ImageNet生成分类器引导样本（仅限前10个类别）
            guidance_scales_subset = [0.0, 1.0, 3.0]
            original_num_classes = model.num_classes
            model.num_classes = min(10, original_num_classes)  # 临时限制类别数
            
            generate_classifier_guided_samples(
                model=model,
                dataset=dataset,
                device=device,
                epoch=epochs-1,
                writer=writer,
                save_dir=save_dir,
                guidance_scales=guidance_scales_subset,
                samples_per_class=2,
                channels=model.out_channels,
                image_size=model.input_image_size
            )
            
            model.num_classes = original_num_classes  # 恢复原始类别数
            
    except Exception as e:
        print(f"最终样本生成失败: {e}")
    
    writer.close()
    
    print(f"模型保存在: {save_dir}")
    print(f"TensorBoard日志: {log_dir}")

def main_train_cifar10(training_mode, use_classifier_guidance):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 128
    epochs = 100
    vae_epochs = 15  # CIFAR-10 VAE训练轮次
    diffusion_epochs = epochs - vae_epochs
    save_dir = "/root/tf-logs/cifar10"
    guidance_scales = [0.0, 1.0, 3.0, 5.0]

    model = stable_diffusion_for_cifar10(use_classifier_guidance=use_classifier_guidance)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CIFAR-10 Stable Diffusion模型参数数量: {total_params:,}")
    print(f"使用Classifier Guidance: {use_classifier_guidance}")
    print(f"设备: {device}")
    
    latent_shape = model.get_encoder_output_shape((1, 3, 32, 32))
    latent_size = latent_shape[2]
    
    dataset = CIFAR10Dataset(
        root_dir="./data/cifar10",
        image_size=model.input_image_size,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        use_augmentation=True,
        normalize_to_neg_one_to_one=True,
        data_percentage=100.0,
    )
    
    learning_rate = 2e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if 'classifier' not in name],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    classifier_optimizer = None
    classifier_lr = 1e-3
    classifier_weight = 1.0
    
    if use_classifier_guidance and model.classifier is not None:
        classifier_optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=classifier_lr,
            weight_decay=weight_decay
        )
    
    # 只为扩散阶段设置调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=diffusion_epochs, eta_min=1e-6
    )
    
    classifier_scheduler = None
    if classifier_optimizer is not None:
        classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=diffusion_epochs, eta_min=1e-6
        )
    
    dataset_info = dataset.get_dataset_info()
    
    training_config = {
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
        "vae_epochs": vae_epochs,
        "diffusion_epochs": diffusion_epochs,
        "save_dir": save_dir,
        "dataset_name": "CIFAR-10",
        "model_name": "CIFAR10_SD_Classifier" if use_classifier_guidance else "CIFAR10_SD",
        
        "optimizer_type": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_scheduler": True,
        "scheduler_type": "CosineAnnealingLR",
        
        "use_classifier_guidance": use_classifier_guidance,
        "classifier_lr": classifier_lr if use_classifier_guidance else None,
        "classifier_weight_decay": weight_decay if use_classifier_guidance else None,
        "guidance_scale": guidance_scales,
        "classifier_weight": classifier_weight,
        
        "total_params": total_params,
        "input_image_size": model.input_image_size,
        "latent_channels": model.latent_channels,
        "latent_size": latent_size,
        "num_timesteps": model.diffusion_model.num_timesteps,
        "use_conditional_generation": model.num_classes is not None,
        "num_classes": model.num_classes,
        
        **dataset_info
    }
    
    print_training_parameters(training_config)

    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = os.path.join(save_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard日志保存到: {log_dir}")
    
    config_text = "\n".join([f"{k}: {v}" for k, v in training_config.items()])
    writer.add_text("Training_Config", config_text, 0)
    
    print(f"\n开始CIFAR-10两阶段训练...\n")
    
    train_two_stage_model(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        save_dir=save_dir,
        writer=writer,
        vae_epochs=vae_epochs,
        diffusion_epochs=diffusion_epochs,
        classifier_optimizer=classifier_optimizer,
        classifier_scheduler=classifier_scheduler,
        classifier_weight=classifier_weight,
        guidance_scales=guidance_scales,
        gradient_clip=1.0,
        dataset_name='CIFAR-10'
    )
    
    print("\n训练完成! 生成最终样本...")
    
    generate_and_log_samples(
        model=model,
        dataset=dataset,
        device=device,
        epoch=epochs-1,
        writer=writer,
        save_dir=save_dir,
        batch_size=10,
        channels=model.out_channels,
        image_size=model.input_image_size
    )
    
    if use_classifier_guidance:
        generate_classifier_guided_samples(
            model=model,
            dataset=dataset,
            device=device,
            epoch=epochs-1,
            writer=writer,
            save_dir=save_dir,
            guidance_scales=guidance_scales,
            samples_per_class=2,
            channels=model.out_channels,
            image_size=model.input_image_size
        )
    
    writer.close()
    
    print(f"模型保存在: {save_dir}")
    print(f"TensorBoard日志: {log_dir}")

if __name__ == "__main__":
    import sys
    
    # 可以通过命令行参数选择数据集类型
    dataset_type = 'mnist'  # 默认使用MNIST
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['imagenet', 'mnist', 'cifar10']:
            dataset_type = sys.argv[1].lower()
    
    print(f"使用数据集: {dataset_type.upper()}")
    
    if dataset_type == 'mnist':
        # MNIST训练模式 - 现在使用两阶段训练
        try:
            print("Starting MNIST two-stage training...")
            main_train(training_mode="two_stage", use_classifier_guidance=False)
            print("MNIST two-stage training completed successfully.")
        except Exception as e:
            print(f"Error in MNIST two-stage training: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            print("Starting MNIST two-stage training with classifier guidance...")
            main_train(training_mode="two_stage", use_classifier_guidance=True)
            print("MNIST two-stage training with classifier guidance completed successfully.")
        except Exception as e:
            print(f"Error in MNIST two-stage training with classifier guidance: {e}")
            import traceback
            traceback.print_exc()
    
    elif dataset_type == 'imagenet':
        # ImageNet训练模式 
        if not IMAGENET_AVAILABLE:
            print("错误: ImageNet模块未找到，无法进行ImageNet训练")
            print("请确保 stablediffusion_imagenet.py 和 dataprocess_imagenet.py 文件存在")
            exit(1)
        
        # 配置1: 小规模快速实验 (推荐开始用这个)
        print("开始ImageNet小规模两阶段训练...")
        try:
            main_train_imagenet(
                image_size=64,              # 64x64分辨率
                num_classes=100,            # 使用前100个类别
                subset_classes=list(range(100)),  # 明确指定使用前100个类别
                data_percentage=10.0,       # 使用10%的数据
                training_mode="two_stage",  # 两阶段训练
                use_classifier_guidance=False,
                model_size="small"          # 使用小模型
            )
        except Exception as e:
            print(f"ImageNet小规模实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    elif dataset_type == 'cifar10':
        # CIFAR-10训练模式
        if not CIFAR10_AVAILABLE:
            print("错误: CIFAR-10模块未找到，无法进行CIFAR-10训练")
            exit(1)

        print("\n开始CIFAR-10两阶段训练（无引导）...")
        try:
            main_train_cifar10(training_mode="two_stage", use_classifier_guidance=False)
            print("CIFAR-10两阶段训练完成。")
        except Exception as e:
            print(f"CIFAR-10两阶段训练失败: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n开始CIFAR-10两阶段训练（带分类器引导）...")
        try:
            main_train_cifar10(training_mode="two_stage", use_classifier_guidance=True)
            print("CIFAR-10两阶段训练（带分类器引导）完成。")
        except Exception as e:
            print(f"CIFAR-10两阶段训练（带分类器引导）失败: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"未知的数据集类型: {dataset_type}")
        print("支持的数据集: mnist, imagenet, cifar10")
        print("使用方法: python train_eval.py [mnist|imagenet|cifar10]")