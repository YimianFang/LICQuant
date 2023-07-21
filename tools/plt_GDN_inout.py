import matplotlib.pyplot as plt
import torch
import os
import numpy as np

### Settings ###
bins = 100
filepath = "savedata/LSQ_GDN_inout"
filelist = os.listdir(filepath)
filelist.sort()
print(filelist)

### Plt ###
fig, axes = plt.subplots(2, 1, figsize=(20, 16), dpi=200)
for filename in filelist:
    if 'in' in filename:
        input = torch.load("savedata/LSQ_GDN_inout/" + filename)
        output = torch.load("savedata/LSQ_GDN_inout/" + filename.replace('in', 'out'))
        _, c, _, _ = input.shape
        input = input.permute(1, 0, 2, 3).reshape(c, -1).cpu()
        output = output.permute(1, 0, 2, 3).reshape(c, -1).cpu()
        if not os.path.exists("savedata/LSQ_GDN_inout_fig/" + filename.replace('.in.pth.tar', '')):
            os.mkdir("savedata/LSQ_GDN_inout_fig/" + filename.replace('.in.pth.tar', ''))
        print("=== Begin to process", filename.replace('.in.pth.tar', ''), "===")
        for ch in range(c):
            axes[0].cla(); axes[1].cla()
            counts, bins = np.histogram(np.array(input[ch]), bins)
            axes[0].stairs(counts, bins)
            # axes[0].hist(input[ch].view(-1)[:1000].cpu(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            axes[0].set_title("Input", fontsize=20)
            axes[0].tick_params(axis='x', labelsize=14)
            axes[0].tick_params(axis='y', labelsize=14)
            counts, bins = np.histogram(np.array(output[ch]), bins)
            axes[1].stairs(counts, bins)
            # axes[1].hist(output[ch].view(-1)[:1000].cpu(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            axes[1].set_title("Output", fontsize=20)
            axes[1].tick_params(axis='x', labelsize=14)
            axes[1].tick_params(axis='y', labelsize=14)
            # axes.legend(loc='best')
            plt.xlabel('Channel Distribution Range', fontsize=20)
            # plt.ylabel('Loss')
            # 设置坐标轴刻度、标签
            # plt.xticks(size=15)
            # plt.yticks(size=15)
            # axes.set_ylim((0, 1.2e-5))
            # axes.set_xticklabels(['zhouyi', 'xuweijia', 'lurenchi', 'chenxiao', 'weiyu', 'guhaiyao'])
            # 设置title
            plt.suptitle('Result of ' + filename.replace('.in.pth.tar', '') + ' in Channel ' + str(ch), fontsize=25)
            # # 网格线
            # axes.grid(linewidth=0.5, which="major", axis='y')
            # # 隐藏上、右边框
            # axes.spines['top'].set_visible(False)
            # axes.spines['right'].set_visible(False)
            plt.show()
            plt.savefig("savedata/LSQ_GDN_inout_fig/" + filename.replace('.in.pth.tar', '') + "/ch" + "_" + str(ch) + ".png", bbox_inches='tight', dpi=fig.dpi) #,pad_inches=0.0