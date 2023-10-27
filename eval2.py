import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import argparse
from loguru import logger
import torch.nn.functional as F
import time
from datasets import CaptionDataset
from utils import get_eval_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def evaluate(args, dataloader,tokenizer, model):
    model.eval()
    for xunhuan in range(2):
        if xunhuan == 0:
            print('using real changeflag')
            # continue
        elif xunhuan == 1:
            print('using predicted changeflag')
            # continue

        beam_size = args.beam_size
        Caption_End = False

        hypotheses = []
        references = []
        nochange_hypotheses = []
        nochange_references = []
        change_hypotheses = []
        change_references = []
        correct = 0  # torch.zeros(1).squeeze().cuda()
        changeflag_correct_nochange = 0
        total = 0
        with torch.no_grad():
            for i, (ori_img, changeflag, caps, mask, caplens, all_captions) in enumerate(tqdm(dataloader, desc=" EVALUATING AT BEAM SIZE " + str(args.beam_size))):
                # if i>20:
                #     break
                if (i + 1) % 5 != 0:
                    continue
                k = beam_size
                ori_img = ori_img.to(device, dtype=torch.float32)
                all_captions = all_captions.tolist()
                if args.dataset_name == 'LEVIR_CC':
                    changeflag = changeflag.to(device)

                # Encode
                if args.dataset_name == 'LEVIR_CC':
                    if xunhuan==0:
                        Sim_cls_AB, pre_flag, inputs_embeds = model.Image_Encoder(changeflag, ori_img)#encoder(image)  # (-1, model.prefix_len, model.prefix_size)
                        pred_changeflag = torch.argmax(pre_flag, 1)
                    elif xunhuan==1:
                        Sim_cls_AB, pre_flag, inputs_embeds = model.Image_Encoder(changeflag, ori_img)
                        pred_changeflag = torch.argmax(pre_flag, 1)
                        _, _, inputs_embeds = model.Image_Encoder(pred_changeflag, ori_img)
                else:
                    inputs_embeds = model.Image_Encoder(ori_img)
                inputs_embeds_dim = inputs_embeds.size(-1)
                num_pixels = inputs_embeds.size(1)
                # We'll treat the problem as having a batch size of k, where k is beam_size
                inputs_embeds = inputs_embeds.expand(k, num_pixels, inputs_embeds_dim)

                # Tensor to store top k sequences' scores; now they're just 0
                top_k_scores = torch.zeros(k, 1).to(device)
                # Lists to store completed sequences and scores
                complete_seqs = []
                complete_seqs_scores = []

                # Start decoding
                step = 1
                fe = inputs_embeds
                while True:
                    # GPT
                    if model.decoder_mode == 'gpt2':
                        out = model.gpt_decoder(inputs_embeds=inputs_embeds)
                        # logits = out.logits  # model.lm_head(out.logits)
                        # logits = model.lm_head(out.logits)
                        out = out.logits
                        if xunhuan==0:
                            logits, pre = model.dual_branch_func(pred_changeflag, out)
                        elif xunhuan==1:
                            logits, pre = model.dual_branch_func(pred_changeflag, out)

                    next_token_logits = logits[:, -1, :]  # 取最后一个单词的预测分布
                    vocab_size = logits.size(-1)  # 50257
                    filtered_logits = next_token_logits
                    scores = F.log_softmax(filtered_logits, dim=-1) # TODO:LSTM:F.log_softmax(scores, dim=1)??
                    # next_token_ids = torch.argmax(scores, dim=1).tolist()

                    # top_k_scores: [s, 1]
                    scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]

                    # For the first step, all k points will have the same scores (since same k previous words, h, c)
                    if step == 1:
                        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                    else:
                        # Unroll and find top scores, and their unrolled indices
                        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                    # Convert unrolled indices to actual indices of scores
                    prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor') # (s)
                    next_word_inds = top_k_words % vocab_size  # (s)
                    # Add new words to sequences
                    if step == 1:
                        seqs = next_word_inds.unsqueeze(1)
                    else:
                        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                    # Which sequences are incomplete (didn't reach <end>)?
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != tokenizer.encode('.')[0]]
                    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        Caption_End = True
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)  # reduce beam length accordingly
                    # Proceed with incomplete sequences
                    if k == 0:
                        break

                    seqs = seqs[incomplete_inds]

                    inputs_embeds = inputs_embeds[prev_word_inds[incomplete_inds]]
                    fe = fe[prev_word_inds[incomplete_inds]]
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                    k_prev_words_embeds = model.gpt_decoder.transformer.wte(k_prev_words).to(device)
                    inputs_embeds = torch.cat((inputs_embeds, k_prev_words_embeds), dim=1)
                    # Break if things have been going on too long
                    if step > 50:
                        # complete_seqs.extend(seqs[incomplete_inds].tolist())
                        # complete_seqs_scores.extend(top_k_scores[incomplete_inds])
                        break
                    step += 1

                changeflag_buff = changeflag.clone()
                prediction = torch.argmax(pre_flag, 1)

                correct += (prediction == changeflag_buff).sum().float()
                total += len(changeflag_buff)
                acc_str = (correct / total) * 100


                changeflag_buff_nochange = changeflag.clone()
                changeflag_buff_nochange[changeflag_buff_nochange > 0.5] = 2
                changeflag_correct_nochange += (prediction == changeflag_buff_nochange).sum().float()
                changeflag_acc_nochange = (changeflag_correct_nochange / total) * 100


                # choose the caption which has the best_score.
                if (len(complete_seqs_scores) ==0):
                    Caption_End = True
                    complete_seqs.extend(seqs[incomplete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[incomplete_inds])
                if (len(complete_seqs_scores) > 0):
                    assert Caption_End
                    # Hypotheses
                    guiyi_complete_seqs_scores = complete_seqs_scores
                    for num_ind in range(len(complete_seqs_scores)):
                        guiyi_complete_seqs_scores[num_ind] = complete_seqs_scores[num_ind]/len(complete_seqs[num_ind])
                    indices = complete_seqs_scores.index(max(guiyi_complete_seqs_scores))
                    seq = complete_seqs[indices]

                    hypotheses.append([w for w in seq if w not in {tokenizer.encode('.')[0]}])

                    # References
                    end_value = tokenizer.encode('.')[0]
                    for one_batch_refs in all_captions:
                        for j in range(len(one_batch_refs)):
                            one_ref = one_batch_refs[j]
                            if end_value in one_ref:
                                end_index = one_ref.index(end_value)
                                ref_remo = one_ref[:end_index]
                            one_batch_refs[j] = ref_remo
                        references.append(one_batch_refs)

                    assert len(references) == len(hypotheses)

                    # change or nochange?
                    if changeflag[0] == 0:
                        nochange_references.append(references[-1])
                        nochange_hypotheses.append(hypotheses[-1])
                    elif changeflag[0] == 1:
                        change_references.append(references[-1])
                        change_hypotheses.append(hypotheses[-1])

            print('len(nochange_references):',len(nochange_references))
            print('len(change_references):', len(change_references))
            if len(nochange_references)>0:
                metrics = get_eval_score(nochange_references, nochange_hypotheses)
            if len(change_references) > 0:
                metrics = get_eval_score(change_references, change_hypotheses)

            metrics = get_eval_score(references, hypotheses)
            print("acc_str:", acc_str)
            print('nochange_acc_nochange:', changeflag_acc_nochange * 2)
            print('change_acc:', acc_str * 2 - changeflag_acc_nochange * 2)

            if xunhuan == 1:
                return metrics

def main(args):
    # tokenizer
    gpt2_type = 'gpt2'
    # gpt2_type = r'C:\Users\lcy\.cache\huggingface\hub\models--gpt2\snapshots\e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    filename = os.listdir(args.model_path)
    for i in range(len(filename)):
        if 'epoch' in filename[i] or 'pth' not in filename[i]:
            continue
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        model_path = os.path.join(args.model_path, filename[i])
        print("model_name:", model_path)
        checkpoint = torch.load(model_path, map_location=args.device)
        model = checkpoint['model_GPT']
        model.eval()

        # load dataset
        dataloader = torch.utils.data.DataLoader(
            CaptionDataset(args.clip_model_type, args.dataset_name, args.data_folder, args.data_name, 'TEST',
                           prefix_length=args.prefix_length),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        logger.info('start predicting')

        metrics = evaluate(args, dataloader, tokenizer, model)

        print("BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
              (metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
               metrics["Bleu_4"],
               metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))
        print("\n")
        print("\n")
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataset_name = "LEVIR_CC"
    parser.add_argument('--dataset_name', default=dataset_name)
    parser.add_argument('--data_folder', default="./data/" + dataset_name,
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default=dataset_name + "_5_cap_per_img", help='base name shared by data files.')

    parser.add_argument('--model_path', default='./checkpoints/cap_model/')#./checkpoints/train_1_method_10/3-times/
    parser.add_argument('--clip_model_type', default="ViT-B/32")#, choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32')
    parser.add_argument('--prefix_length', type=int, default=59)  # 7*7+10
    parser.add_argument('--prompt_len', type=int, default=5)
    parser.add_argument('--uni_prompt_1_len', type=int, default=5)
    parser.add_argument('--len_change_emmbed', type=int, default=1)


    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8, help='for data-loading')


    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # prefix_len
    args.prefix_length = 2 * (img_size[0] * img_size[1]) + args.prompt_len + args.len_change_emmbed


    main(args)
