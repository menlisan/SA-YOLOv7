import os
import random
import time
import argparse
import numpy as np
import torch

from models.gradcam import YOLOV7GradCAM, YOLOV7GradCAMPP
from models.yolov7_object_detector import YOLOV7TorchObjectDetector
import cv2
# 数据集中的类别名(与标签数字相对应)
names = ['capacitors', 'resistors', 'inductors', 'diodes', 'ICs','transistors']
#capacitors橘色 resistor黄色 inductors粉 diodes紫色 ICs蓝色transistors湖绿
colors = {'capacitors': [0,140,255], 'resistors': [0,255,255] , 'inductors': [197,181,255], 'diodes': [255,32,131],'ICs':[255,118,72] , 'transistors': [255,255,0]}



# yolov7网络中，detect层前的三层输出
target_layers = ['102_act', '103_act', '104_act']  # yolov7

#target_layers = ['101_act']  # yolov7
# target_layers = ['74_act', '75_act', '76_act']  # yolov7-tiny

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="runs/train_linux/yolov7_last_2.pt", help='Path to the model')#runs/train_linux/exp914/weights/best.pt
parser.add_argument('--img-path', type=str, default='myImg\\3\\9.jpg', help='input image path')
parser.add_argument('--output-dir', type=str, default='runs\\cam', help='output dir')
parser.add_argument('--img-size', type=int, default=320, help="input image size")
parser.add_argument('--target-layer', type=str, default='76_act',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam method: gradcam, gradcampp')
parser.add_argument('--device', type=str, default='gpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')
parser.add_argument('--no_text_box', action='store_true',
                    help='do not show label and box on the heatmap')
args = parser.parse_args()


def get_res_img(bbox,mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = (res_img / 255).astype(np.float32)
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat



def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    # cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    # img = cv2.imread('temp.jpg')

    # Plots one bounding box on image img
    tl = round(line_thickness/2) or round(0.002 * (img.shape[0] + img.shape[1]) / 2) - 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf )[0]
        outside = c1[1] - t_size[1] - 3 >= 0  # label fits outside box up
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if outside else c1[1] + t_size[1] + 3
        outsize_right = c2[0] - img.shape[:2][1] > 0  # label fits outside box right
        c1 = c1[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c1[0], c1[1]
        c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2 if outside else c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)
    return img

# 检测单个图片
def main(img_path):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    device = torch.device('cuda:0' if args.device == 'gpu' else 'cpu')
    input_size = (args.img_size, args.img_size)
    # 读入图片
    img = cv2.imread(img_path)  # 读取图像格式：BGR
    print('[INFO] Loading the model')
    # 实例化YOLOv7模型，得到检测结果
    model = YOLOV7TorchObjectDetector(args.model_path, device, img_size=input_size, names=names)
    # img[..., ::-1]: BGR --> RGB
    # (480, 640, 3) --> (1, 3, 480, 640)
    torch_img = model.preprocessing(img[..., ::-1])
    tic = time.time()
    # 遍历三层检测层

    for u,target_layer in enumerate(target_layers):
        # 获取grad-cam方法
        if args.method == 'gradcam':
            saliency_method = YOLOV7GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif args.method == 'gradcampp':
            saliency_method = YOLOV7GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  # 得到预测结果
        print('masks:',masks)
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        # 保存设置
        imgae_name = os.path.basename(img_path)  # 获取图片名
        save_path = f'{args.output_dir}{imgae_name[:-4]}/{args.method}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f'[INFO] Saving the final image at {save_path}')
        res_img = result.copy()
        mask =torch.zeros(masks[0].shape,device='cuda:0')
        for i, m in enumerate(masks):
            # if(torch.count_nonzero(m).item() != 0 and i != 0):
            #     mask = torch.add(mask,m)
            # # 获取目标的热力图
            m = torch.where(torch.isnan(m), torch.full_like(m, 0), m)
            if i != 0:
                mask.add_(m,alpha=1)
        mask = mask.squeeze(0).mul(255).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
                    np.uint8)
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        # n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
        n_heatmat = (heatmap / 255).astype(np.float32)
        res_img = (res_img / 255).astype(np.float32)
        cv2.imwrite(f'ut/n_heatmat_{u}.jpg', (n_heatmat * 255).astype(np.uint8))
        cv2.imwrite(f'ut/res_img.jpg', (res_img * 255).astype(np.uint8))
        res_img = cv2.addWeighted(res_img,0.5, n_heatmat,0.5,0)
        res_img = (res_img / res_img.max())
        # cv2.imwrite(f'ut/temp.jpg', (res_img * 255).astype(np.uint8))
        # res_img = cv2.imread(f'ut/temp.jpg')
        # output_path = f'{save_path}/{target_layer[:-4]}_{i}t.jpg'
        # cv2.imwrite(output_path, res_img)
        res_img = cv2.imread(f'ut/p2/test2.jpg')

        for i, mask in enumerate(masks):
            # 获取目标的位置和类别信息
            bbox, cls_name = boxes[0][i], class_names[0][i]
            label = f'{cls_name} {conf[0][i]}'  # 类别+置信分数
            res_img = plot_one_box(bbox, res_img, label=label, color=colors[int(names.index(cls_name))],
                                   line_thickness=3)
            print(f'{imgae_name[:-4]}_{target_layer[:-4]}.jpg done!!')
        # 缩放到原图片大小
        res_img = cv2.resize(res_img, dsize=(img.shape[:-1][::-1]))
        output_path = f'{save_path}/{target_layer[:-4]}_m.jpg'
        cv2.imwrite(output_path, res_img)
    print(f'Total time : {round(time.time() - tic, 4)} s')

def plot_box(x, img, color,line_thickness=10):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    # cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    # img = cv2.imread('temp.jpg')
    # Plots one bounding box on image img
    tl = round(0.002 * (img.shape[0] + img.shape[1]))  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    print(f'c1：{c1} c2:{c2}\n')
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # tf = max(tl - 2, 1)  # font thickness
    # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf )[0]
    # outside = c1[1] - t_size[1] - 3 >= 0  # label fits outside box up
    # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if outside else c1[1] + t_size[1] + 3
    # outsize_right = c2[0] - img.shape[:2][1] > 0  # label fits outside box right
    # c1 = c1[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c1[0], c1[1]
    # c2 = c2[0] - (c2[0] - img.shape[:2][1]) if outsize_right else c2[0], c2[1]
    # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    return img

# 检测单个图片
def main2(img_path):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    device = torch.device('cuda:0' if args.device == 'gpu' else 'cpu')
    input_size = (args.img_size, args.img_size)
    # 读入图片
    img = cv2.imread(img_path)  # 读取图像格式：BGR
    print(f'imgshape:{img.shape}\n')
    print('[INFO] Loading the model')
    # 实例化YOLOv7模型，得到检测结果
    model = YOLOV7TorchObjectDetector(args.model_path, device, img_size=input_size, names=names)
    # img[..., ::-1]: BGR --> RGB
    # (480, 640, 3) --> (1, 3, 480, 640)
    torch_img = model.preprocessing(img[..., ::-1])
    tic = time.time()
    # 遍历三层检测层

    for u,target_layer in enumerate(target_layers):
        # 获取grad-cam方法
        if args.method == 'gradcam':
            saliency_method = YOLOV7GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif args.method == 'gradcampp':
            saliency_method = YOLOV7GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  # 得到预测结果
        if len(masks) == 0:
            continue
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        # 保存设置
        imgae_name = os.path.basename(img_path)  # 获取图片名
        save_path = f'{args.output_dir}/{args.method}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        images=[result]
        heat_map_num = 0  # 新添加
        heat_map_sum = 0  # 新添加
        for i, mask in enumerate(masks):
            res_img = result.copy()
            bbox, cls_name = boxes[0][i], class_names[0][i]
            res_img, heat_map = get_res_img(bbox, mask, res_img)
            heat_map_num += 1  # 新添加
            heat_map_sum += heat_map  # 新添加 1)把masks中的所有的热力图叠加起来
            # label = f'{cls_name}'  # 类别+置信分数
            res_img = plot_box(bbox, res_img, color=colors[int(names.index(cls_name))],
                                   line_thickness=5)
            images.append(res_img)
        # 2）将叠加结果归一化到[0，1]
        heat_map_avg = heat_map_sum / (heat_map_num + 1e-5)
        heat_map_avg = (heat_map_avg / heat_map_avg.max())

        single_res_img = result.copy()
        single_res_img = single_res_img / 255
        # 3）然后再和原图叠加
        # single_res_img = cv2.add(single_res_img, heat_map_avg)
        single_res_img = cv2.addWeighted(single_res_img, 0.4,heat_map_avg,0.9,0)
        single_res_img = (single_res_img / single_res_img.max())
        # 4）最后再将结果转换到到[0,255]
        single_res_img = single_res_img * 255
        single_res_img = single_res_img.astype(np.uint8)
        single_output_path = f'{ save_path}/{target_layer[:-4]}_{imgae_name}'
        cv2.imwrite(single_output_path, single_res_img)
    print(f'Total time : {round(time.time() - tic, 4)} s')

if __name__ == '__main__':
    # 图片路径为文件夹
    if os.path.isdir(args.img_path):
        img_list = os.listdir(args.img_path)
        print(img_list)
        for item in img_list:
            # 依次获取文件夹中的图片名，组合成图片的路径
            main2(os.path.join(args.img_path, item))
    # 单个图片
    else:
        main2(args.img_path)

