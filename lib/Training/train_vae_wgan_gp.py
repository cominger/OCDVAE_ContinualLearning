import time
import torch
import copy
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy
from lib.Utility.visualization import visualize_confusion
from lib.Utility.visualization import visualize_image_grid
import lib.OpenSet.meta_recognition as mr
from .augmentation import blur_data

def train(Dataset, model, criterion, epoch, optimizer, writer, device, save_path, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and log_weights (bool).
    """

    # Create instances to accumulate losses etc.
    class_losses = AverageMeter()
    recon_losses = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    G_losses_dec = AverageMeter()
    G_losses_enc = AverageMeter()
    G_losses_fake = AverageMeter()

    D_losses = AverageMeter()
    D_losses_real = AverageMeter()
    D_losses_fake = AverageMeter()
    D_gp_losses = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # train
    for i, (inp, target) in enumerate(Dataset.train_loader):
        inp = inp.to(device)
        target = target.to(device)

        recon_target = inp
        class_target = target

        # this needs to be below the line where the reconstruction target is set
        # sample and add noise to the input (but not to the target!).
        # if args.denoising_noise_value > 0.0:
        #     noise = torch.randn(inp.size()).to(device) * args.denoising_noise_value
        #     inp = inp + noise

        # if args.blur:
        #     inp = blur_data(inp, args.patch_size, device)

        # measure data loading time
        data_time.update(time.time() - end)

        # Model explanation: Conventionally GAN architecutre update D first and G
        #### D Update#### 
        mu_label = None

        # update Real image
        real_z = model.module.forward_D(recon_target, mu_label)
        D_loss_real = - torch.mean(real_z)                                 #WGAN-GP
        D_losses_real.update(D_loss_real.item(), inp.size(0))

        # update Recon Image
        class_samples, recon_samples, mu, std = model(inp)
        n,b,c,x,y = recon_samples.shape
        recon_z = model.module.forward_D((recon_samples.view(n*b,c,x,y)).detach(), mu_label)
        # fake_samples = model.module.generate(recon_target.size(0))
        # fake_z = model.module.forward_D(fake_samples.detach(), mu_label)

        # D_loss_fake = (torch.mean(recon_z) + torch.mean(fake_z)) * 0.5    #WGAN-GP
        D_loss_fake = torch.mean(recon_z)                                   #WGAN-GP
        # D_loss_fake = torch.mean(fake_z)                                    #WGAN-GP

        D_losses_fake.update(D_loss_fake.item(), inp.size(0))

        GAN_D_loss = (D_loss_real + D_loss_fake)  

        # Compute loss for gradient penalty
        alpha = torch.rand(recon_target.size(0),1,1,1).to(device)
        # x_hat = (alpha * recon_target.data + (1-alpha) * fake_samples.data).requires_grad_(True)
        x_hat = (alpha * recon_target + (1-alpha) * recon_samples.view(n*b,c,x,y)).requires_grad_(True)
        out_x_hat = model.module.forward_D(x_hat, mu_label)
        D_loss_gp = model.module.discriminator.gradient_penalty(out_x_hat, x_hat)
        D_gp_losses.update(D_loss_gp, inp.size(0))

        GAN_D_loss += args.lambda_gp*D_loss_gp

        D_losses.update(GAN_D_loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer['enc'].zero_grad()
        optimizer['dec'].zero_grad()
        optimizer['disc'].zero_grad()
        GAN_D_loss.backward()
        optimizer['disc'].step()

        #### G Update####
        if i % 1 == 0:
            #update encoder
            mu, std = model.module.encode(inp)
            recon_z = model.module.reparameterize(mu, std)
            class_samples = model.module.classifier(recon_z)
            recon_samples = model.module.decode(recon_z)
            class_samples = torch.unsqueeze(class_samples,dim=0)
            recon_samples = torch.unsqueeze(recon_samples,dim=0)
            # class_samples, recon_samples, mu, std = model(inp)
            mu_label = None

            # OCDVAE calculate loss
            class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
                                                         device, args)

            enc_loss = args.var_cls_beta * class_loss + \
                    args.l1_weight * recon_loss + \
                    args.var_beta * kld_loss

            output = torch.mean(class_samples, dim=0)
            # record precision/accuracy and losses
            prec1 = accuracy(output, target)[0]
            top1.update(prec1.item(), inp.size(0))
            
            class_losses.update(class_loss.item(), inp.size(0))
            recon_losses.update(recon_loss.item(), inp.size(0))
            kld_losses.update(kld_loss.item(), inp.size(0))

            G_losses_enc.update(enc_loss.item(), inp.size(0))

            optimizer['enc'].zero_grad()
            enc_loss.backward()
            optimizer['enc'].step()
            
            #update decoder
            recon_samples = model.module.decode(recon_z.detach())
            recon_samples = torch.unsqueeze(recon_samples,dim=0)
            _, recon_loss, _ = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
                                                         device, args)
            n,b,c,x,y = recon_samples.shape
            recon_z = model.module.forward_D((recon_samples.view(n*b,c,x,y)), mu_label)

            # GAN_gen losses
            GAN_G_loss = - torch.mean(recon_z)
            G_losses_fake.update(GAN_G_loss.item(), inp.size(0))

            # total decoder update GAN + recon
            dec_loss = args.l1_weight * recon_loss + \
                    args.var_gan_weight * GAN_G_loss
            G_losses_dec.update(dec_loss.item(), inp.size(0))

            optimizer['dec'].zero_grad()
            optimizer['disc'].zero_grad()
            dec_loss.backward()
            optimizer['dec'].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if i % args.print_freq == 0:
            print ("OCD_VAELoss: ")
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                  'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                   epoch+1, i, len(Dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, cl_loss=class_losses, top1=top1,
                   recon_loss=recon_losses, KLD_loss=kld_losses))
            print ("GANLoss: ")
            print('G Loss {G_loss.val:.4f} ({G_loss.avg:.4f})\t'
                  'D Loss {D_loss.val:.4f} ({D_loss.avg:.4f})'.format(G_loss = G_losses_dec, D_loss=D_losses))

        if (i == (len(Dataset.train_loader) - 2)) and (epoch % args.visualization_epoch == 0):
            visualize_image_grid(inp, writer, epoch + 1, 'train_input_snapshot', save_path)
            visualize_image_grid(recon_samples.view(n*b,c,x,y), writer, epoch + 1, 'train_reconstruction_snapshot', save_path)

    # TensorBoard summary logging
    writer.add_scalar('training/train_precision@1', top1.avg, epoch)
    writer.add_scalar('training/train_average_loss', losses.avg, epoch)
    writer.add_scalar('training/train_KLD', kld_losses.avg, epoch)
    writer.add_scalar('training/train_class_loss', class_losses.avg, epoch)
    writer.add_scalar('training/train_recon_loss', recon_losses.avg, epoch)
    # Discriminator related
    writer.add_scalar('training/train_D_loss', D_losses.avg, epoch)
    writer.add_scalar('training/train_D_loss_real', D_losses_real.avg, epoch)
    writer.add_scalar('training/train_D_loss_fake', D_losses_fake.avg, epoch)
    writer.add_scalar('training/train_D_loss_gp', D_gp_losses.avg, epoch)
    # Generator related
    writer.add_scalar('training/train_G_loss_enc', G_losses_enc.avg, epoch)
    writer.add_scalar('training/train_G_loss_fake', G_losses_fake.avg, epoch)
    writer.add_scalar('training/train_G_loss_dec', G_losses_dec.avg, epoch)
    

    # If the log weights argument is specified also add parameter and gradient histograms to TensorBoard.
    if args.log_weights:
        # Histograms and distributions of network parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins="auto")
            # second check required for buffers that appear in the parameters dict but don't receive gradients
            if value.requires_grad and value.grad is not None:
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins="auto")

    print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}\t'
          ' GAN: Generator {G_losses.avg:.5f} Discriminator {D_losses.avg:.4f}'
        .format(loss=losses, top1=top1, G_losses=G_losses_dec, D_losses=D_losses))
