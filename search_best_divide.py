import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from vis import *
import torchvision
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
import numpy as np
from model import multi_model
import time
import PIL
import dataset
from metric import *
from torch.utils.data.dataloader import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def search_for_best_p(prob, label, step=0.001, title="None"):
    n = len(label)
    label = np.array(label)
    p = 0
    best_acc = 0
    final_p = 0
    new_true = np.zeros_like(label)
    while p < 1.0:
        p += step
        for i in range(n):
            if prob[i] < p:
                new_true[i] = 0
            else:
                new_true[i] = 1
        now_acc = sum(label == new_true) / n
        if now_acc >= best_acc:
            best_acc = now_acc
            final_p = p
    for i in range(n):
        if prob[i] < final_p:
            new_true[i] = 0
        else:
            new_true[i] = 1
    draw_cm(label, new_true, "best_p_{}".format(title), [0, 1], "./result")
    draw_roc(label, prob, "best_p_roc_{}".format(title), "./result")
    return final_p, best_acc, new_true


trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([256, 256]),
                            transforms.Normalize(mean=0.5, std=0.5)])


task_type = "VEGF"
# task_type = "PDL1"
if task_type == "PDL1":
    model = torch.load("../result/weights/PDL1/test-fold1-AUC0.7100-ACC0.5000.pth")
    # model = torch.load("../result/weights/PDL1/test-fold2-AUC0.6900-ACC0.5000.pth")
    # model = torch.load("../result-xiaorong/PDL1/test-fold2-AUC0.7600-ACC0.5000.pth")
else:
    model = torch.load("../result/weights/VEGF/test-fold3-AUC0.8500-ACC0.7500.pth")
    # model = torch.load("../result/weights/VEGF/test-fold2-AUC0.7000-ACC0.5000.pth")
    # model = torch.load("../result-xiaorong/VEGF/test-fold3-AUC0.7100-ACC0.5000.pth")
print(task_type)
# test_data = dataset.CCA_dataset('{}'.format(task_type), "./{}_test.txt".format(task_type))
test_data = dataset.CCA_dataset('{}'.format(task_type), "./{}_val_fold3.txt".format(task_type))
# test_data = dataset.CCA_dataset('{}'.format(task_type), "./{}_val_fold3.txt".format(task_type))
test_loader = DataLoader(test_data, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
model.eval()
prob_result = []
label_true = []
result_csv = open("{}_result_val.csv".format(task_type), "w")
result_csv.write("id,p0,p1\n")
for data, p_id in test_loader:
    f1, f2, DWI, TW1, TW2, t_label = data
    f1, f2, t_label = f1.to(device), f2.to(device), t_label.to(device)
    _, output = model(f1, f2, DWI, TW1, TW2)
    label_true.append(t_label.cpu().item())
    output = output.cpu().detach().numpy()
    prob_result.append(output[0][1])
    # print(p_id[0])
    result_csv.write("{},{:.4f},{:.4f}\n".format(p_id[0], output[0][0], output[0][1]))
result_csv.close()
p, acc, now_true = search_for_best_p(prob_result, label_true, title=task_type)
print("divd:{:.4f},ACC:{:.4f}".format(p, acc))
data_type_list = ['acc', 'auc', 'f1', 'precision', 'sensitivity', 'specificity']
lower = []
upper = []
for datas in data_type_list:
    l, u = bootstrap_data(label_true, prob_result, [0, 1], data_type=datas, p=p)
    lower.append(l)
    upper.append(u)
vali = Metrics(label_true, now_true, prob_result)
ACC = vali.accuracy()
print("ACC: {:.4f}, {:.4f}-{:.4f}".format(ACC, lower[0], upper[0]))
AUC = vali.AUC()
print("AUC: {:.4f}, {:.4f}-{:.4f}".format(AUC, lower[1], upper[1]))
F1 = vali.f1()
print("f1: {:.4f}, {:.4f}-{:.4f}".format(F1, lower[2], upper[2]))
precision = vali.precision()
print("precision: {:.4f}, {:.4f}-{:.4f}".format(precision, lower[3], upper[3]))
sensitivity = vali.sensitivity()
print("sensitivity: {:.4f}, {:.4f}-{:.4f}".format(sensitivity, lower[4], upper[4]))
specificity = vali.specificity()
print("specificity: {:.4f}, {:.4f}-{:.4f}".format(specificity, lower[5], upper[5]))

"""
VEGF
0.559 0.8
ACC: 0.8000, 0.7700-0.8300
AUC: 0.8400, 0.8097-0.8691
f1: 0.8000, 0.7680-0.8294
precision: 0.8000, 0.7667-0.8346
sensitivity: 0.8000, 0.7560-0.8400
specificity: 0.8000, 0.7560-0.8400
PDL1
0.522 0.7
ACC: 0.7000, 0.6660-0.7340
AUC: 0.7100, 0.6696-0.7491
f1: 0.7000, 0.6653-0.7339
precision: 0.7000, 0.6654-0.7375
sensitivity: 0.7000, 0.6520-0.7480
specificity: 0.7000, 0.6520-0.7440
"""


# prob = np.array([0.18, 0.4, 0.2, 0.5, 0.3, 0.7, 0.8, 0.1, 0.2, 0.14])
# label = [0, 1, 0, 1, 1, 1, 1, 0, 0, 1]
#
# p, acc = search_for_best_p(prob, label)
# print(p, acc)
