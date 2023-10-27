import time
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import GPT2Tokenizer
from datasets import CaptionDataset
from utils import *

from models_CC import LEVIR_CC_CaptionModel
from eval2 import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def train(args, train_loader, Caption_model, Caption_model_optimizer,Caption_model_lr_scheduler, epoch):
    """
    Performs one epoch's training.
    """
    Caption_model.train()
    Caption_model_optimizer.zero_grad(set_to_none=True)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    losses_cap = AverageMeter()
    pre_flag_losses = AverageMeter()

    cap_top5accs = AverageMeter()  # top5 accuracy
    # flag_top5accs = AverageMeter()

    start = time.time()
    # Batches
    correct = 0
    total = 0
    accum_steps = 3
    for idx, (ori_img, changeflag, area, caps, mask, caplens) in enumerate(train_loader):
        # if idx ==10:
        #     break
        data_time.update(time.time() - start)

        # Move to GPU, if available
        ori_img = ori_img.to(device, dtype=torch.float32)
        if args.dataset_name == 'LEVIR_CC':
            changeflag = changeflag.to(device)

        caps = caps.to(device)
        mask = mask.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        if args.dataset_name == 'LEVIR_CC':
            Sim_cls_AB, pre_flag, outputs = Caption_model(caps, changeflag, ori_img, mask)


        logits = outputs[:, args.prefix_length - 1: -1]
        # mask_loss = mask[:, args.prefix_length:]
        # logits= logits*mask_loss.unsqueeze(-1)
        scores = logits.reshape(-1, logits.shape[-1])
        targets = caps.flatten()
        loss_cap = F.cross_entropy(scores, targets, ignore_index=0)

        # pre_flag_loss
        changeflag_buff = changeflag.clone()
        pre_flag_loss = F.cross_entropy(pre_flag, changeflag_buff)  # , ignore_index=0

        # total loss
        loss = loss_cap
        loss = loss / accum_steps

        # Back prop.
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(Caption_model_optimizer, args.grad_clip)

        # Update weights
        if (idx + 1) % accum_steps == 0 or (idx + 1) == len(train_loader):
            Caption_model_optimizer.step()
            Caption_model_lr_scheduler.step()
            Caption_model_optimizer.zero_grad(set_to_none=True)

        # Keep track of metrics
        cap_top5 = accuracy(scores, targets, 3)
        # flag_top = accuracy(pre_flag, changeflag_buff, 1)

        prediction = torch.argmax(pre_flag, 1)
        correct += (prediction == changeflag_buff).sum().float()
        total += len(changeflag_buff)
        acc_str = (correct / total)*100

        losses.update(loss.item(), args.batch_size)
        losses_cap.update(loss_cap.item(), args.batch_size)
        pre_flag_losses.update(pre_flag_loss.item(), args.batch_size)

        cap_top5accs.update(cap_top5)
        # flag_top5accs.update(flag_top)
        batch_time.update(time.time() - start)

        start = time.time()
        if idx % args.print_freq == 0:
            print("Epoch:{}/{} step:{}/{} Loss:{:.4f} AVG_Loss:{:.4f} CAP_Loss:{:.4f} CAP_AVG_Loss:{:.4f} Flag_Loss:{:.4f} Flag_AVG_Loss:{:.4f} CAP_Acc:{:.2f} Flag_Acc:{:.2f} Batch_time:{:.2f}s"
                  .format(epoch+0, args.epochs, idx+0, len(train_loader), losses.val, losses.avg, losses_cap.val, losses_cap.avg,
                          pre_flag_losses.val, 0,
                          cap_top5accs.val, acc_str, batch_time.val))

def main(args):

    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU

    # Initialize
    if args.clip_model_type == 'RN50':
        clip_emb_dim = 1024
        img_feature_dim, img_size = 2048, [7, 7]
    elif args.clip_model_type == 'RN101':
        clip_emb_dim = 512
        img_feature_dim, img_size = 2048, [7, 7]
    elif args.clip_model_type == 'RN50x4':
        clip_emb_dim = 640
        img_feature_dim, img_size = 2560, [9, 9]
    elif args.clip_model_type == 'RN50x16':
        clip_emb_dim = 768
        img_feature_dim, img_size = 3072, [12, 12]
    elif args.clip_model_type == 'ViT-B/16' or args.clip_model_type == 'ViT-L/16':
        clip_emb_dim = 512
        img_feature_dim, img_size = 768, [14, 14]
    elif args.clip_model_type == 'ViT-B/32' or args.clip_model_type == 'ViT-L/32':
        clip_emb_dim = 512
        img_feature_dim, img_size = 768, [7, 7]

    # prefix_length = h*w+prompt_len
    if args.dataset_name == 'LEVIR_CC':
        args.prefix_length = 2*(img_size[0] * img_size[1]) + args.prompt_len + args.len_change_emmbed

    print("Train both prefix and GPT") if args.finetune_gpt2 else print("Train only prefix")

    Caption_model = LEVIR_CC_CaptionModel(encoder_mode=args.clip_model_type, decoder_mode=args.decoder_mode,
                                           prompt_len=args.prompt_len, uni_prompt_1_len=args.uni_prompt_1_len,
                                        len_change_emmbed=args.len_change_emmbed,
                                           img_feature_dim=img_feature_dim, img_feature_h=img_size[0],
                                           img_feature_w=img_size[1],
                                           num_layers=args.num_layers)

    Caption_model.set_finetune(args.finetune_gpt2)

    Caption_model_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, Caption_model.parameters()),
                                         lr=args.model_lr)
    Caption_model_lr_scheduler = StepLR(Caption_model_optimizer, step_size=900, gamma=1)

    # Move to GPU, if available
    Caption_model = Caption_model.to(device)
    print("Checkpoint_savepath:{}".format(args.savepath))


    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.clip_model_type, args.dataset_name, args.data_folder, args.data_name, 'TRAIN',prefix_length=args.prefix_length),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,persistent_workers=True)


    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.clip_model_type, args.dataset_name, args.data_folder, args.data_name, 'VAL',
                       prefix_length=args.prefix_length),
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True,persistent_workers=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # Decay learning rate if there is no improvement for x consecutive epochs
        if epochs_since_improvement == args.stop_criteria:
            print("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            adjust_learning_rate(Caption_model_optimizer, 0.7)

        # One epoch's training
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

        train(args,
              train_loader=train_loader,
              Caption_model=Caption_model,
              Caption_model_optimizer=Caption_model_optimizer,
              Caption_model_lr_scheduler=Caption_model_lr_scheduler,
              epoch=epoch)
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        # continue
        # One epoch's validation
        # gpt2_type = 'gpt2'
        gpt2_type = r'C:\Users\lcy\.cache\huggingface\hub\models--gpt2\snapshots\e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        metrics = evaluate(args, test_loader, tokenizer, Caption_model)

        recent_bleu4 = metrics["Bleu_4"]#+metrics["Bleu_3"]+metrics["Bleu_2"]+metrics["Bleu_1"]
        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        metrics = {'Bleu_4': 0}
        checkpoint_name = args.clip_model_type.replace('/', '_')
        save_checkpoint(args, checkpoint_name, epoch, epochs_since_improvement, Caption_model,
                       Caption_model_optimizer, metrics, is_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Change_Captioning')
    # Data parameters
    dataset_name = "LEVIR_CC"  # LEVIR_CC
    parser.add_argument('--dataset_name', default=dataset_name)
    parser.add_argument('--data_folder', default="./data/"+dataset_name,help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default=dataset_name+"_5_cap_per_img",help='base name shared by data files.')

    # Model parameters
    parser.add_argument('--clip_model_type', default="ViT-B/32")#, choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32')
    parser.add_argument('--decoder_mode', default="gpt2", choices=('gpt2', 'transformer_decoder'))

    parser.add_argument('--finetune_gpt2', action='store_true', default=False)

    parser.add_argument('--prefix_length', type=int, default=59) #7*7+10
    parser.add_argument('--prompt_len', type=int, default=5)
    parser.add_argument('--uni_prompt_1_len', type=int, default=5)
    parser.add_argument('--len_change_emmbed', type=int, default=1)

    parser.add_argument('--num_layers', type=int, default=23, help="2 layers for fisrt transformer_encoder and "
                                                                   "3 layers for second transformer_encoder")
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=12, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=4, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--model_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')

    # FIXME:用于验证集合
    parser.add_argument('--Split', default="TEST", help='which')
    parser.add_argument('--beam_size', type=int, default=3, help='beam_size.')
    parser.add_argument('--savepath', default="./checkpoints/3-times/")

    args = parser.parse_args()

    main(args)
