from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
import numpy as np
from model import multi_model
import time
import PIL


def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    # if titles == False:
    #     titles = "0123456789"
    for i in range(1, lens + 1):
        plt.xticks(())
        plt.yticks(())
        plt.subplot(1, 3, i)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        if i == lens:
            plt.figtext(0.5, 0.025, titles[-1], ha='center', fontsize=17)
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    # plt.legend()
    # plt.savefig(fname, bbox_inches='tight')
    plt.show()


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()  # [0]
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    # np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr


# path = "./image_data/PD_L1_data/0/T2WI/image/P23/1.npy"
# path = "./image_data/PD_L1_data/0/T1WI/image/P63/12.npy"
# path = "./image_data/PD_L1_data/0/DWI/image/P12/2.npy"

path = "./image_data/PD_L1_data/1/T2WI/image/P34/3.npy"
# path = "./image_data/PD_L1_data/1/DWI/image/P34/3.npy"
# path = "./image_data/PD_L1_data/1/T1WI/image/P34/3.npy"

# path = "./image_data/PD_L1_data/0/T1WI/image/P61/10.npy"

trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([256, 256]),
                            transforms.Normalize(mean=0.5, std=0.5)])
imgg = np.load(path)
plt.imshow(imgg, cmap='gray')
plt.title("Input normal image, start to calculate...")
plt.show()
img = trans(imgg)
img = img.unsqueeze(0)
img = img.float()
input_tensors = img
# model = torch.load("../result/weights/PDL1/test-fold1-AUC0.7100-ACC0.5000.pth", map_location='cpu')
# self.DWI_model = resnet(classes=64, model_type=18)
# self.TW1_model = resnet(classes=64, model_type=18)
# self.TW2_model = resnet(classes=64, model_type=18)
# target_layers = [model.TW1_model.model.layer4[-1]]  # 如果传入多个layer，cam输出结果将会取均值
# with GradCAM(model=model.TW1_model.model, target_layers=target_layers) as cam:
#     # targets = [ClassifierOutputTarget(386), ClassifierOutputTarget(386)]  # 指定查看class_num为386的热力图
#     # aug_smooth=True, eigen_smooth=True 使用图像增强是热力图变得更加平滑
#     grayscale_cams = cam(input_tensor=input_tensors)  # targets=None 自动调用概率最大的类别显示
#     for grayscale_cam, tensor in zip(grayscale_cams, input_tensors):
#         # 将热力图结果与原图进行融合
#         rgb_img = tensor2img(tensor)
#         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#         myimshows([rgb_img, grayscale_cam, visualization], ["image", "cam", "image + cam"])
# target_layers = [model.TW2_model.model.layer4[-1]]  # 如果传入多个layer，cam输出结果将会取均值
# with GradCAM(model=model.TW2_model.model, target_layers=target_layers) as cam:
#     # targets = [ClassifierOutputTarget(386), ClassifierOutputTarget(386)]  # 指定查看class_num为386的热力图
#     # aug_smooth=True, eigen_smooth=True 使用图像增强是热力图变得更加平滑
#     grayscale_cams = 1 - cam(input_tensor=input_tensors)  # targets=None 自动调用概率最大的类别显示
#     for grayscale_cam, tensor in zip(grayscale_cams, input_tensors):
#         # 将热力图结果与原图进行融合
#         rgb_img = tensor2img(tensor)
#         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#         myimshows([rgb_img, grayscale_cam, visualization], ["image", "cam", "image + cam"])
# target_layers = [model.DWI_model.model.layer4[-1]]  # 如果传入多个layer，cam输出结果将会取均值
# with GradCAM(model=model.DWI_model.model, target_layers=target_layers) as cam:
#     # targets = [ClassifierOutputTarget(386), ClassifierOutputTarget(386)]  # 指定查看class_num为386的热力图
#     # aug_smooth=True, eigen_smooth=True 使用图像增强是热力图变得更加平滑
#     grayscale_cams = 1 - cam(input_tensor=input_tensors)  # targets=None 自动调用概率最大的类别显示
#     for grayscale_cam, tensor in zip(grayscale_cams, input_tensors):
#         # 将热力图结果与原图进行融合
#         rgb_img = tensor2img(tensor)
#         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#         myimshows([rgb_img, grayscale_cam, visualization], ["image", "cam", "image + cam"])
# https://blog.csdn.net/a486259/article/details/123905702
