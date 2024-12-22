import json
import os
import shutil

'''
작성일 : 2023 - 12- 27
작성자 : 신재호
xml_to_yolo_bbox -> yolo용 xywh좌표로 변환하는 함수
label_maker -> jsonfile을 읽어서, yolo좌표 txt파일을 생성하는 함수
class_dist_checker -> 라벨들의 분포를 확인하는 코드
image_mover -> 라벨에 맞춰서 이미지를 저장하는 함수
'''


def xml_to_yolo_bbox(bbox, b_height, b_widht):
    x_center = ((bbox[1][0] + bbox[0][0]) / 2)
    y_center = ((bbox[1][1] + bbox[0][1]) / 2)
    width = (bbox[1][0] - bbox[0][0])
    height = (bbox[1][1] - bbox[0][1])
    return [x_center/b_widht, y_center/b_height, width/b_widht, height/b_height]


def label_maker(file_path, save_path, label_dict):
    for file in os.listdir(file_path):
        data = json.load(open(file_path + "/" + file, "r"))
        # {'version': '4.5.9', 'flags': {},'shapes': [{'label': 'Styrofoam','points': [[1056, 1106], [1770, 1491]],'origin': 'coast','group_id': None,'shape_type': 'rectangle','flags': {}}], ...}
        name = data["imagePath"][:-4]
        # FD_8030
    
        temp = []
        for i in range(len(data["shapes"])):
            label = data["shapes"][i]["label"]
            height = data["imageHeight"]
            width = data["imageWidth"]
            
            la = label_dict[label]
            # 딕셔너리의 라벨을 넣어, yolo용 라벨을 반환
            
            xywh = xml_to_yolo_bbox(data["shapes"][i]["points"], height, width)
            temp_1 = [la] + xywh
            # [la, x_center/b_widht, y_center/b_height, width/b_widht, height/b_height]
            temp_1 = ' '.join(str(num) for num in temp_1)
            # "la x_center/b_widht y_center/b_height width/b_widht height/b_height"
            temp.append(temp_1)
            
            file_name = f'{save_path}{name}.txt'
            # txt파일 저장경로 지정
            
            with open(file_name, 'w') as file:
                for t in temp:
                    file.write(t + '\n')

def class_dist_checker(label_path, class_num):
    count = {f"{i}" : 0 for i in range(class_num+1)}
    for txt in os.listdir(label_path):
        with open(label_path + txt, "r") as f:
            for line in f:
                count[line[0]] += 1
    print("총 이미지 개수 :", len(os.listdir(label_path)))
    print("클래스분포 :", count)

def image_mover(txt_label_path, image_path, save_path):
    temp = []
    for txt in os.listdir(txt_label_path):
        temp.append(txt[:-4] + ".jpg")
        
    for t in temp:
        source_path = os.path.join(image_path, t)
        target_path = os.path.join(save_path, t)
        shutil.copy(source_path, target_path)


