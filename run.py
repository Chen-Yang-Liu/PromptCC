import os
import time
import json
import pynvml

if __name__ == '__main__':

    start_time = time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time()))
    print(start_time)

    str = [
        # "python ./train.py --num_layers 23 --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --savepath ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/1-times/",
        # "python ./train.py --num_layers 23 --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --savepath ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/2-times/",
        # "python ./train.py --num_layers 23 --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --savepath ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/3-times/",

        "python ./eval2.py --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --model_path ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/1-times/",
        "python ./eval2.py --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --model_path ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/2-times/",
        # "python ./eval2.py --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --model_path ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/3-times/",
        # "python ./eval2.py --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --model_path ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/4-times/",
        # "python ./eval2.py --prompt_len 5 --uni_prompt_1_len 5 --len_change_emmbed 1 --model_path ./checkpoints2/xiaorong/prompt_51_num_layers_23_ignore0_hand_craft_prompt/5-times/",
    ]

    k = 0

    kkkkkk = 0
    while (k < len(str)):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free = meminfo.free / 1024 ** 2  # 9 * 1024
        print(free)

        if free > (19 * 1024):
            # kkkkkk =kkkkkk+1
            # if kkkkkk == 1:
            #     print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            #     time.sleep(300)
            # if kkkkkk>1:
            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            os.system(str[k])
            print('*************************************************************\n \n')
            k = k + 1
        else:
            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            time.sleep(700)
        # print(meminfo.total / 1024 ** 2)  # 总的显存大小
        # print(meminfo.used / 1024 ** 2)  # 已用显存大小
        # print(meminfo.free / 1024 ** 2)  # 剩余显存大小