import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
import argparse
import torch.nn.functional as F
import time
import skimage.io as io
from PIL import Image
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def evaluate(args, ori_img,tokenizer, model):
    model.eval()
    for xunhuan in range(2):
        if xunhuan == 0:
            continue
            print('using real changeflag')
        elif xunhuan == 1:
            # continue
            print('using predicted changeflag')


        beam_size = args.beam_size
        Caption_End = False

        hypotheses = []

        correct = 0  # torch.zeros(1).squeeze().cuda()
        changeflag_correct_nochange = 0
        total = 0
        with torch.no_grad():
            k = beam_size
            ori_img = ori_img.to(device, dtype=torch.float32)
            # all_captions = all_captions.tolist()
            if args.dataset_name == 'LEVIR_CC':
                # changeflag = changeflag.to(device)
                changeflag = torch.zeros((1)).to(device)

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
                    out = out.logits
                    if xunhuan==0:
                        logits, pre = model.dual_branch_func(pred_changeflag, out)
                    elif xunhuan==1:
                        logits, pre = model.dual_branch_func(pred_changeflag, out)
                elif model.decoder_mode == 'transformer_decoder':
                    # DecoderTransformer
                    out = model.DecoderTransformer(memory=fe,tgt=inputs_embeds)
                    logits = model.lm_head(out).permute(1, 0, 2)

                next_token_logits = logits[:, -1, :]  # 取最后一个单词的预测分布
                vocab_size = logits.size(-1)  # 50257
                # topk filter
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

                sent = ''
                caption = tokenizer.convert_ids_to_tokens(hypotheses[-1])

                for st in caption:
                    sent = sent + str(st)
                # # print('\n')
                # print(i,":  ",sent.replace('Ġ', ' '))
                # print('\n')
                return sent


def main(args, img2cap):
    # 分词器
    # gpt2_type = 'gpt2'
    gpt2_type = r'C:\Users\lcy\.cache\huggingface\hub\models--gpt2\snapshots\e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    filename = os.listdir(args.model_path)
    for i in range(len(filename)):
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        model_path = os.path.join(args.model_path, filename[i])
        print("model_name:", model_path)
        checkpoint = torch.load(model_path, map_location=args.device)
        model = checkpoint['model_GPT']

        model.eval()
        # 加载数据集
        # file = open(r"F:\LCY\Prompt_unify_captioning_1026\output\output_test.txt", "w", encoding='utf-8')
        result_json = {}
        refer_id_list = []
        hypo_id_list = []
        refer_id_list_map = []
        hypo_id_list_map = []


        # Load word map (word2ix)
        word_map_file = 'F:\LCY\TGRS_RSICCFormer\Change_Captioning_Transformer\data\WORDMAP_LEVIR_CC_5_cap_per_img_5_min_word_freq.json'
        with open(word_map_file, 'r') as f:
            word_map = json.load(f)

        kkk = -1
        for name in img2cap.keys():
            kkk = kkk +1
            # if kkk >100:
            #     break
            image_path = os.path.join(args.image_path, name)

            path1 = image_path#r'E:\Dataset\Caption\change_caption\Levir_CC_dataset\images\train\A\train_000333.png'
            path2 = image_path.replace('A','B')#r'E:\Dataset\Caption\change_caption\Levir_CC_dataset\images\train\B\train_000333.png'
            ori_img_A = io.imread(path1)
            ori_img_B = io.imread(path2)

            clip_model, preprocess = clip.load(args.clip_model_type, device='cuda', jit=False)
            A = preprocess(Image.fromarray(ori_img_A)).unsqueeze(0)
            B = preprocess(Image.fromarray(ori_img_B)).unsqueeze(0)
            ori_img = (torch.cat([A, B], dim=0)).unsqueeze(0).to(device)
            caption = evaluate(args, ori_img, tokenizer, model)
            # for cap in caption:
            cap = caption

            print('\n', path1)

            result_json[name] = {}
            hypotheses = cap.replace('Ġ',' ')
            references = img2cap[name].copy()
            result_json[name]['hypotheses'] = hypotheses
            result_json[name]['references'] = references
            print("hypotheses:", hypotheses)
            print("references:", references)
            # 先转换为id再计算指标
            # METHOD1 用GPT的分词器
            hypo_id = tokenizer.encode(hypotheses)
            refer_id = [tokenizer.encode(raw) for raw in references]
            hypo_id_list.append(hypo_id)
            refer_id_list.append(refer_id)
            # result_json[name]['metrics'] = get_eval_score([refer_id], [hypo_id])
            # METHOD2 用之前的词汇表
            hypo_id_map = [word_map[word] if word in word_map.keys() else word_map["<unk>"] for word in hypotheses.split()]
            refer_id_map = [[word_map[word] if word in word_map.keys() else word_map["<unk>"] for word in raw.split()] for raw in references]
            hypo_id_list_map.append(hypo_id_map)
            refer_id_list_map.append(refer_id_map)
            # result_json[name]['metrics_wordmap'] = get_eval_score([refer_id_map], [hypo_id_map])


        # time.sleep(10)
        # file.close()
        print(result_json)
        # metric = get_eval_score(refer_id_list, hypo_id_list)
        # metric_map = get_eval_score(refer_id_list_map, hypo_id_list_map)
        # print(metric)
        # print(metric_map)
        out_path = r'./eval_results_fortest'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open( os.path.join(out_path, 'promptCC_traindataset.json'), 'w') as f:
            json.dump(result_json, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataset_name = "LEVIR_CC"
    parser.add_argument('--dataset_name', default=dataset_name)
    parser.add_argument('--data_name', default=dataset_name + "_5_cap_per_img", help='base name shared by data files.')

    parser.add_argument('--image_path', default=r'D:\Dataset\Caption\change_caption\Levir_CC_dataset\images\test\A')
    parser.add_argument('--model_path', default='F:/LCY/Prompt-CC/Prompt_unify_captioning_1026/checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hard/2-times/')#./checkpoints/train_1_method_10/3-times/
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--prefix_length', type=int, default=59)  # 7*7+10
    parser.add_argument('--prompt_len', type=int, default=5)
    parser.add_argument('--uni_prompt_1_len', type=int, default=5)
    parser.add_argument('--len_change_emmbed', type=int, default=1)

    parser.add_argument('--Split', default="TRAIN", help='which for validation')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4,
                        help='for data-loading; right now, only 0 works with h5pys in windows.')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_path = r'D:\Dataset\Caption\change_caption\Levir_CC_dataset\LevirCCcaptions_v0.json'
    # 打开json文件
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    # 新建一个字典，用于存放img_path和caption
    img2cap = {}
    for img in data['images']:
        if img['split'] != args.Split.lower():
            continue
        captions = []
        for c in img['sentences']:
            if len(c['tokens']) <= 50:
                sent = ''
                for w in c['tokens']:
                    sent = sent +' '+ w
                captions.append(sent[1:].replace(' .','').replace('.',''))
        img2cap[img['filename']] = captions

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
    elif args.clip_model_type == 'ViT-B/16':
        clip_emb_dim = 512
        img_feature_dim, img_size = 768, [14, 14]
    elif args.clip_model_type == 'ViT-B/32':
        clip_emb_dim = 512
        img_feature_dim, img_size = 768, [7, 7]

    args.prefix_length = 2 * (img_size[0] * img_size[1]) + args.prompt_len + args.len_change_emmbed


    main(args, img2cap)
