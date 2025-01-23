import argparse
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import os
import cv2


class BBoxObj:
    def __init__(self, json_dict, class_names, image_size, target_names, not_using_labels, name_dict):
        self.xmin = json_dict['bbox'][0]
        self.ymin = json_dict['bbox'][1]
        self.xmax = json_dict['bbox'][0] + json_dict['bbox'][2]
        self.ymax = json_dict['bbox'][1] + json_dict['bbox'][3]

        self.im_h, self.im_w = image_size
        self.xmin = max(self.xmin, 0)
        self.ymin = max(self.ymin, 0)
        self.xmax = min(self.xmax, self.im_w)
        self.ymax = min(self.ymax, self.im_h)

        self.class_num = json_dict['label_id']
        self.class_name = class_names[json_dict['label_id']]
        if self.class_name in name_dict:
            self.class_name = name_dict[self.class_name]
        if self.class_name not in target_names or self.class_name in not_using_labels:
            if self.class_name not in not_using_labels:
                not_using_labels.append(self.class_name)
                logger.info('not using:', not_using_labels)
            self.class_num = -1
        else:
            self.class_num = target_names.index(self.class_name)
        # self.dontcare = dontcare

    @property
    def bbox(self):
        return int(self.xmin), int(self.ymin), int(self.xmax), int(self.ymax)

    @property
    def cvat_bbox(self):
        return (self.xmin + self.xmax) / self.im_w / 2, (self.ymin + self.ymax) / self.im_h / 2, self.width / self.im_w, self.height / self.im_h

    @property
    def width(self):
        xmin, ymin, xmax, ymax = self.bbox
        return (xmax - xmin)

    @property
    def height(self):
        xmin, ymin, xmax, ymax = self.bbox
        return (ymax - ymin)

    @property
    def area(self):
        return self.width * self.height


class DatumaroTaskParser:
    def __init__(self, json_fpath, target_names, not_using_labels, name_dict):
        values = json.load(open(json_fpath))
        class_names = values['categories']['label']['labels']
        class_names = [d['name'] for d in class_names]
        self.image_fpaths = []
        self.annotations = []
        folder = Path(json_fpath).stem
        image_dir = os.path.dirname(json_fpath) + '/../images/%s' % folder
        for anno_id in range(len(values['items'])):
            image_name = values['items'][anno_id]['image']['path']
            image_size = values['items'][anno_id]['image']['size']
            annos = values['items'][anno_id]['annotations']
            self.image_fpaths.append(image_dir + '/' + image_name)
            self.annotations.append([BBoxObj(anno, class_names, image_size, target_names, not_using_labels, name_dict) for anno in annos])
        self.cvat_id_label_mapping = self.generate_cvat_label_id_mapping(values)

    def get_image(self, idx):
        return cv2.imread(self.image_fpaths[idx])

    def generate_cvat_label_id_mapping(self, values):
        cvat_id_label_mapping = {}
        for idx, d in enumerate(values["categories"]["label"]["labels"]):
            cvat_id_label_mapping[idx] = d["name"]
        return cvat_id_label_mapping

    @property
    def cvat_label_id_mapping(self):
        return self.cvat_id_label_mapping

    @property
    def count(self):
        return len(self.image_fpaths)

target_names = ['head']

mapping_dict = {
    'head': 'head'
}
not_using_labels = ['credit_card', 'basket']

TO_DELETE_NAME = "to_delete"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default='/data3/dataset/key_motion/extract/31', help="the sorce file path")
    parser.add_argument("--dst_path", type=str, default='/home/michaelhe/data/hand_object', help="the target file path")
    parser.add_argument("--testing_interval", type=int, default=5, help="the testing data interval")
    parser.add_argument("--remove_olds", action="store_true", help="whether to remove old data")
    parser.add_argument("--soft_link", action="store_true", help="whether to use soft link; if source images are in local SSD, recommanded to use soft-link for preparing speed. If source images are on NAS or cloud or HDD, or if it is necessary to do image resizing or modification image on data-preparing, set it False")
    parser.add_argument("--hard_link_for_images_contains_deleted", action="store_true", help="whether to link image with hard link when contains deleted happend")
    parser.add_argument("--not_using_labels", type=str, default=",".join(not_using_labels), help="not using labels, seperated by ',' ex:--not_using_labels credit_card,basket")
    parser.add_argument("--target_names", type=str, default=",".join(target_names), help="target names list, seperated by ',' ex:--not_using_labels credit_card,basket")
    parser.add_argument("--mapping_dict_path", type=str, default=None, help="the path to the mapping_dict")
    args = parser.parse_args()
    args.not_using_labels = args.not_using_labels.split(",")
    args.target_names = args.target_names.split(",")
    if args.mapping_dict_path is not None:
        mapping_dict = json.load(open(args.mapping_dict_path))

    json_paths = sorted(list(Path(args.src_path).rglob('*.json')))
    cnt = 0
    for json_path in json_paths:
        task_parser = DatumaroTaskParser(json_path.as_posix(), args.target_names, args.not_using_labels, mapping_dict)
        cnt += 1
        if cnt % args.testing_interval == 0:
            task_type = 'eval'
        else:
            task_type = 'train'
        task_id = json_path.parent.parent.name  # ex: /data3/dataset/key_motion/extract/31/task_id/***.json
        dst_dir = os.path.join(args.dst_path, task_type, task_id)
        os.makedirs(dst_dir, exist_ok=True)
        for idx in range(task_parser.count):
            image_name = os.path.basename(task_parser.image_fpaths[idx])
            annos = task_parser.annotations[idx]
            dst_im_fpath = dst_dir + '/' + image_name
            anno_fpath = dst_dir + '/' + image_name.replace('.png', '.txt').replace('.jpg', '.txt')
            contains_to_delete = False
            with open(anno_fpath, 'w') as fstream:
                for obj in task_parser.annotations[idx]:
                    if obj.class_name == TO_DELETE_NAME:
                        contains_to_delete = True
                        continue
                    if obj.class_num == -1:
                        continue
                    fstream.write(f"{obj.class_num} {obj.cvat_bbox[0]} {obj.cvat_bbox[1]} {obj.cvat_bbox[2]} {obj.cvat_bbox[3]}\n")
            if args.soft_link and not (args.hard_link_for_images_contains_deleted and contains_to_delete):
                os.symlink(task_parser.image_fpaths[idx], dst_im_fpath)
            else:
                if not os.path.isfile(dst_im_fpath):
                    im = cv2.imread(task_parser.image_fpaths[idx])
                    for obj in task_parser.annotations[idx]:
                        if obj.class_name == TO_DELETE_NAME:
                            im[int(obj.ymin):int(obj.ymax), int(obj.xmin):int(obj.xmax), :] = 0
                    cv2.imwrite(dst_dir + '/' + image_name, im)
