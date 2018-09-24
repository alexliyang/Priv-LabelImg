import json
import xml.etree.ElementTree as ET
from lxml import etree
import numpy as np
class Voc_To_Coco():
    def __init__(self, _voc_xml_path, _save_json_path):
        self._voc_xml_path = _voc_xml_path
        self._save_json_path = _save_json_path

        self.images = []
        self.categories = []
        self.annotations = []
        self.bbox=[]
        self.points=[]
        self.vis=[]
        self.keypoints=[]
    def processing_xml(self):
        tree = ET.parse(self._voc_xml_path)
        #images part
        filename_node= tree.findall('filename')
        filename=filename_node[0].text
        size=tree.findall('size')
        height=size[0].findall('height')[0].text
        width=size[0].findall('width')[0].text
        id=0
        image = {'height': float(height), 'width': float(width), 'id': id, 'file_name': filename}
        self.images.append(image)
        #categories
        #annotations
        object = tree.findall('object')

        for i,obj in enumerate(object):
            bbox = obj.find('bndbox')
            label = obj.find('name').text.lower().strip()
            x1 = np.maximum(0.0, float(bbox.find('xmin').text))
            y1 = np.maximum(0.0, float(bbox.find('ymin').text))
            print('e',bbox.find('xmax').text)
            x2 = np.minimum(float(width) - 1.0, float(bbox.find('xmax').text))
            y2 = np.minimum(float(height) - 1.0, float(bbox.find('ymax').text))
            # rectangle = [x1, y1, x2, y2]
            bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]  # [x,y,w,h]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            points = obj.findall('point')
            for i, point in enumerate(points):
                keypoint = point.find('keypoints').text
                vis = point.find('visible').text
                self.points.append(keypoint)
                self.vis.append(vis)
            self.keypoints=self.processing_points(self.points,self.vis)

            annotation = {'segmentation': [], 'iscrowd': 0, 'area': area, 'image_id': id,
                          'bbox': bbox,
                          'category_id': label, 'id': id,'keypoints':self.keypoints}
            self.annotations.append(annotation)
    def processing_points(self,points,vis):
        s=[]
        v=[]
        point_list=[]
        for i,point in enumerate(points):
            for ii in point.strip('[').strip(']').strip().split(','):
                s.append(int(ii.strip()))
        for viss in vis:
            for visss in viss.strip('[').strip(']').strip().split(','):
                v.append(int(visss))

        for i in range(len(v)):
            point_list.append(s[i])
            point_list.append(s[2*i])
            point_list.append(v[i])
        return point_list

    def save_json(self):
        data_coco = {'images': self.images, 'categories': self.categories, 'annotations': self.annotations}
        json.dump(data_coco, open(self._save_json_path, 'w'))

vv=Voc_To_Coco('E:/Annotation/person/3_point.xml','E:/Priv-lab1-2018-9-17-master/3_point.json')
vv.processing_xml()
vv.save_json()
print(vv.keypoints)
