# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import pickle
from tqdm import tqdm


# dl_manager.iter_archive(download_path[2]),

class StanfordDogs():
    def __init__(self):
        self._DOGS_DIR = "/work/ml/stanford-dogs-dataset"
        self._LIST_FILE = ["file_list.mat", "test_list.mat", "train_list.mat"]
        self._ANNOTATIONS_DIR = "Annotation"
        self._IMAGE_DIR = "images"
        self._NAME_RE = re.compile(r"([\w-]*/)*([\w]*.jpg)$")  # res = _NAME_RE.match(fname)

    def split_generators(self, pickle_filename):
        xml_file_list = collections.defaultdict(str)
        train_list, test_list, label_names = [], [], []

        # Parsing the mat file which contains the list of train/test images
        def parse_mat_file(file_name):
            parsed_mat_arr = loadmat(file_name, squeeze_me=True)
            file_list = [element.split("/")[-1] for element in parsed_mat_arr["file_list"]]
            return file_list, parsed_mat_arr

        for fname in os.listdir(self._DOGS_DIR):
            full_file_name = os.path.join(self._DOGS_DIR, fname)
            if "train" in fname:
                train_list, train_mat_arr = parse_mat_file(full_file_name)
                label_names = set([element.split("/")[-2].lower() for element in train_mat_arr["file_list"]])
            elif "test" in fname:  # and 0:
                test_list, _ = parse_mat_file(full_file_name)
        for root, _, files in tqdm(os.walk(os.path.join(self._DOGS_DIR, self._ANNOTATIONS_DIR))):
            # Parsing the XML file which have the image annotations
            for fname in (files):
                annotation_file_name = os.path.join(root, fname)
                with open(annotation_file_name, "rb") as f:
                    xml_file_list[fname] = ET.parse(f)
        pick_data = [
            {
                "archive": "train",
                "file_names": train_list,
                "annotation_files": xml_file_list,
                "label_names": label_names,
            },
            {
                "archive": "test",
                "file_names": test_list,
                "annotation_files": xml_file_list,
            }
        ]
        if os.path.exists(pickle_filename):
            os.remove(pickle_filename)
        with open(pickle_filename, 'wb') as f:
            print(f'saving data to file:{pickle_filename} ...')
            pickle.dump(pick_data, f, protocol=4)
            print(f'done! writing train {len(train_list)} records and test {len(test_list)}.')

    def generate_examples(self, archive, file_names, annotation_files):
        """Generate dog images, labels, bbox attributes given the directory path.
        Args:
          archive: object that iterates over the zip
          file_names : list of train/test image file names obtained from mat file
          annotation_files : dict of image file names and their xml object
        Yields:
          Image path, Image file name, its corresponding label and
          bounding box values
        """
        label_names = self.info.features["label"].names
        bbox_attrib = ["xmin", "xmax", "ymin", "ymax", "width", "height"]

        for fname, fobj in archive:
            res = self._NAME_RE.match(fname)
            if not res or (fname.split("/")[-1] not in file_names):
                continue

            label = res.group(1)[:-1].lower()
            file_name = res.group(2)
            attributes = collections.defaultdict(list)
            for element in annotation_files[file_name.split(".")[0]].iter():
                # Extract necessary Bbox attributes from XML file
                if element.tag.strip() in bbox_attrib:
                    attributes[element.tag.strip()].append(float(element.text.strip()))

            # BBox attributes in range of 0.0 to 1.0
            def normalize_bbox(bbox_side, image_side):
                return min(bbox_side / image_side, 1.0)

            def build_box(attributes, n):
                return tfds.features.BBox(
                    ymin=normalize_bbox(attributes["ymin"][n], attributes["height"][0]),
                    xmin=normalize_bbox(attributes["xmin"][n], attributes["width"][0]),
                    ymax=normalize_bbox(attributes["ymax"][n], attributes["height"][0]),
                    xmax=normalize_bbox(attributes["xmax"][n], attributes["width"][0]),
                )

            yield fname, {
                "image":
                    fobj,
                "image/filename":
                    fname,
                "label":
                    label_names.index(label),
                "objects": [{
                    "bbox": build_box(attributes, n)
                } for n in range(len(attributes["xmin"]))]
            }


def main():
    sd = StanfordDogs()
    sd.split_generators('sdogs.pkl')
    '''
    then we need write more img data ...
    img_path = data['image_name'][i]
    gt_boxs = data['boxes'][i]
    gt_classes = data['gt_classes'][i]
    nobj = data['num_objs'][i]
    bboxs = data['selective_search_boxes'][i]
    nroi = len(bboxs)

    img = Image.open('data/JPEGImages/' + img_path)
    '''


if __name__ == '__main__':
    main()
