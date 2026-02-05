import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm
import cv2
from torch.utils.data import ConcatDataset
from ultralytics import YOLO
from PIL import Image,ImageDraw, ImageFont
import time
import warnings

def video_to_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # 逐帧读取视频并保存为 PNG 文件
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame)

    # 释放视频对象
    cap.release()

def evaluate(model, cropped_ori_image):
        model.eval()
        with torch.no_grad():
            images=np.array(cropped_ori_image)
            images = images.to(device)
            # print(len(images))
            outputs = model(images)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            # print("------------------------------------------")
            print("predicted---------", predicted)

def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def delete_folder_if_exists(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 如果存在，先删除文件夹及其内容
        shutil.rmtree(folder_path)
        print(f"已删除文件夹：{folder_path}")
    else:
        print(f"文件夹不存在：{folder_path}")


def images_to_video(image_folder, output_video_path, fps=30):
    # 获取图像文件列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # 确定第一张图像的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历图像并写入视频
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # 释放视频对象
    video.release()

def clear_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            clear_folder(file_path)
            os.rmdir(file_path)

if __name__ == '__main__':

    # # 输入视频文件路径
    video_path = r"C:\Users\jiang\Desktop\服务外包\视频\10.mp4"

    # 输出视频路径
    output_video_path = (r"C:\Users\jiang\Desktop\10.mp4")

    # # 输出帧的文件夹路径
    output_folder = r"C:\Users\jiang\Desktop\frames"
    final_folder = r"C:\Users\jiang\Desktop\well cover.v1i.yolov8\runs\detect\predict"

    clear_folder(output_folder)
    # 调用帧数函数
    video_to_frames(video_path, output_folder)

    model = YOLO("runs/detect/train3/weights/best.pt")

    # 指定要遍历的文件夹路径
    process_path = (r"C:\Users\jiang\Desktop\frames")

    profiles = list_files_in_directory(process_path)

    # 要检查和删除的文件夹路径
    folder_path_to_delete = r"C:\Users\jiang\Desktop\well cover.v1i.yolov8\runs\detect\predict"

    # 忽略特定类型的警告
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # 调用函数
    delete_folder_if_exists(folder_path_to_delete)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"  # CUDA out of memory. Tried to allocate 110.00 MiB (GPU 0; 11.00 GiB total capacity; 10.09 GiB already allocated; 0 bytes free; 10.23 GiB reserved in total by PyTorch)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为模型期望的尺寸
        # transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载预训练模型
    jjh_model = timm.create_model('vit_base_patch16_224', pretrained=False)
    # # 定义本地权重文件的路径
    local_weights_path = r"C:\Users\jiang\Desktop\well cover.v1i.yolov8\model_weight_epoch_90.pt"

    jjh_model = jjh_model.to(device)

    # 修改分类器
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    # 修改分类器 vit和resnet不一样
    num_ftrs = jjh_model.head.in_features
    jjh_model.head = nn.Linear(num_ftrs, 5)
    jjh_model.load_state_dict(torch.load(local_weights_path))

    frame_count=0
    model1_total_time=0
    model2_total_time=0
    for profile in profiles:
        pro_image = Image.open(profile)
        frame_count += 1
        # ori_response = model.predict(ofile, confidence=40, overlap=0)
        # # 解析 JSON 响应
        # ori_predictions = ori_response.json()['predictions']
        model1_start_time = time.time()
        pro_results = model(source=profile, show=False, save=True)
        model1_end_time = time.time()
        model1_execution_time = model1_end_time-model1_start_time
        model1_total_time += model1_execution_time
        # print(f"YOLO执行时间：{model1_execution_time} 秒")

        print()

        if pro_results[0]:
            if pro_results[0]:
                # print(pro_results[0].probs)
                for box in pro_results[0].boxes.xyxy:
                    pro_predictions = box.tolist()
                    ori_conf = pro_results[0].boxes.conf[0].item()
                    print(ori_conf)
                    cropped_ori_image = pro_image.crop((pro_predictions[0], pro_predictions[1], pro_predictions[2],pro_predictions[3]))

                    model2_start_time = time.time()

                    jjh_model.eval()
                    with torch.no_grad():
                        # images = np.array(cropped_ori_image)
                        images = test_transforms(cropped_ori_image)
                        images = torch.unsqueeze(images,0).to(device)
                        # print(len(images))
                        outputs = jjh_model(images)
                        # print(outputs.data)
                        _, predicted = torch.max(outputs.data, 1)
                        # print("------------------------------------------")
                        predicted = predicted.numpy()
                        print("predicted---------", predicted)

                        model2_end_time = time.time()
                        model2_execution_time = model2_end_time - model2_start_time
                        model2_total_time += model2_execution_time
                        # print(f"MODEL执行时间：{model2_execution_time} 秒")

                        if(predicted == [0]):
                            text="broke"
                            font_color =  (255, 165, 0)
                        elif(predicted == [1]):
                            text="circle"
                            font_color =  (255, 165, 0)
                        elif(predicted == [2]):
                            text="good"
                            font_color =  (0, 255, 0)
                        elif(predicted == [3]):
                            text="lose"
                            font_color = (255, 0, 0)
                        elif(predicted == [4]):
                            text="uncovered"
                            font_color =  (255, 165, 0)


                        # 自定义文本位置
                        custom_position = (pro_predictions[0]+2, pro_predictions[1]+2)  # 自定义位置坐标

                        # 调用函数并传入自定义位置参数


                        frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
                        final_path = os.path.join(final_folder, f"frame_{frame_count:05d}.png")

                        final_image=Image.open(final_path)

                        font_size = 20

                        draw = ImageDraw.Draw(final_image)

                        # 加载字体
                        font = ImageFont.truetype("arial.ttf", font_size)

                        # 计算文本大小
                        text_width, text_height = draw.textsize(text, font=font)

                        # 计算背景矩形的位置和大小
                        bg_x, bg_y = custom_position
                        bg_width = text_width + 10  # 加上一些额外的空间
                        bg_height = text_height + 5  # 加上一些额外的空间
                        bg_color=(255, 255, 255)

                        # 绘制背景矩形
                        draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=bg_color)

                        # 在背景矩形上添加文本
                        text_position = (bg_x + 5, bg_y + 2)  # 稍微偏移以使文本居中



                        draw.text(text_position, text, font=font, fill=font_color)

                        # 保存修改后的图像
                        final_image.save(final_path)


                # 保存修改后的图像
                final_image.save(frame_path)



    # 输入图像文件夹路径
    image_folder = r"C:\Users\jiang\Desktop\frames"

    # 视频帧率（可选，默认为30）
    fps = 24

    # 调用函数
    images_to_video(image_folder, output_video_path, fps)

    # print(f"YOLO平均执行时间：{model1_total_time / frame_count} 秒/帧")
    # print(f"MODEL平均执行时间：{model2_total_time / frame_count} 秒/帧")

