import matplotlib.pyplot as plt
import torch

### Settings ###
layer = [0, 2, 4, 6]
Loss = 1 # {0: "Mean", 1: "RD"}
abs = True

### Plt ###
LossTypeList = ["Mean", "RD"]
LossType = LossTypeList[Loss]
filename = "result_abs" if abs else "result"
re = torch.load("MixQ/"+ filename + "/layer" + str(layer) + ".pth.tar")
re_top = re[LossType + "top"]
re_bottom = re[LossType + "bottom"]
re_rand = re[LossType + "rand"]
x = list(range(1, len(re_top) * 5, 5))
fig, axes = plt.subplots(1, 1, figsize=(14, 7))
# width = 0.1
# 画柱状图
# axes.bar(torch.tensor(range(N0.numel())), quan_w_s["g_a.0.quan_w_fn.s"].view(-1).abs() / 2**12 / N0.abs(), label='LSQ / UN', color="#D2ACA3")
# axes.bar(torch.tensor(range(plttensor.numel())), plttensor.abs(), label='UN', color="#D2ACA3")
plt.plot(x, re_top, 'o-', color = 'palevioletred', label="Top")
plt.plot(x, re_bottom, 'o-', color = 'g', label="Bottom")#o-:圆形
plt.plot(x, re_rand, 'o-', color = 'cornflowerblue', label="Rand")
axes.legend(loc='best')
plt.xlabel('Num of channels quantized with 16 bits')
plt.ylabel('Loss')
# 设置坐标轴刻度、标签
# axes.set_xticks(list(range(N0.size()))+1)
# axes.set_yticks()
# axes.set_ylim((0, 1.2e-5))
# axes.set_xticklabels(['zhouyi', 'xuweijia', 'lurenchi', 'chenxiao', 'weiyu', 'guhaiyao'])
# 设置title
axes.set_title('Result of Layer ' + str(layer) + ' with ' + LossType + ' Loss' )
# 网格线
axes.grid(linewidth=0.5, which="major", axis='y')
# 隐藏上、右边框
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.show()
plt.savefig("MixQ/" + filename + "/layer" + str(layer) + "_" + LossType + ".png", bbox_inches='tight', dpi=fig.dpi) #,pad_inches=0.0