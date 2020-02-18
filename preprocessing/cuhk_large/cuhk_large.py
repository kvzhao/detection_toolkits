import os
import numpy as np
from scipy.io import loadmat

TestG50 = 'TestG50'
TestG100 = 'TestG100'
TestG500 = 'TestG500'
TestG1000 = 'TestG1000'
TestG2000 = 'TestG2000'
TestG4000 = 'TestG4000'


"""
['s14859.jpg']
[[9]]
[[(array([[307,  40, 118, 403]], dtype=uint16), array([[0]], dtype=uint8))
  (array([[503,  50,  89, 396]], dtype=uint16), array([[0]], dtype=uint8))
  (array([[ 66, 135,  98, 251]], dtype=uint8), array([[0]], dtype=uint8))
  (array([[183,  76, 105, 364]], dtype=uint16), array([[0]], dtype=uint8))
  (array([[410, 138,  62, 182]], dtype=uint16), array([[0]], dtype=uint8))
  (array([[463, 139,  40, 165]], dtype=uint16), array([[0]], dtype=uint8))
  (array([[741, 150,  52, 175]], dtype=uint16), array([[0]], dtype=uint8))
  (array([[671, 128,  51, 188]], dtype=uint16), array([[0]], dtype=uint8))
  (array([[597, 148,  46, 168]], dtype=uint16), array([[0]], dtype=uint8))]]
"""

class ImageAnnotation:
    pass


class TestData:
    imname = None #str
    idlocate = None #[l, r, t, b]
    ishard = None #[0 or 1]
    idname = None #str

    def __init__(self, imname, idlocate, ishard, idname):
        self.imname = imname
        if len(idlocate) > 0:
            # for easy use in imread, swap X and Y
            self.idlocate = [int(idlocate[1]), int(idlocate[1])+int(idlocate[3]), int(idlocate[0]), int(idlocate[0])+int(idlocate[2])]
        else:
            self.idlocate = []
        self.ishard = ishard
        self.idname = idname

class CUHK_Large:
    image_path = None

    image_annotation = None
    person_annotation = None
    pool_annotation = None
    train_query = None
    test_query = None
    occlusion_annotation = None
    low_resolution_annotation = None

    def __init__(self, dataset_root, test_query_type = None):
        self.image_path = os.path.join(dataset_root, 'Image/SSM')

        path = os.path.join(dataset_root, 'annotation/Images.mat')
        self.image_annotation = loadmat(path)['Img']

        path = os.path.join(dataset_root, 'annotation/Person.mat')
        self.person_annotation = loadmat(path)['Person']

        path = os.path.join(dataset_root, 'annotation/pool.mat')
        self.pool_annotation = loadmat(path)['pool']

        path = os.path.join(dataset_root, 'annotation/test/train_test/Train.mat')
        self.train_query = loadmat(path)['Train']

        if test_query_type is None:
            test_query_type = TestG100
        path = os.path.join(dataset_root, 'annotation/test/train_test/' + test_query_type + '.mat')
        self.test_query = loadmat(path)[test_query_type]

        path = os.path.join(dataset_root, 'annotation/test/subset/Occlusion.mat')
        self.occlusion_annotation = loadmat(path)['Occlusion1']

        path = os.path.join(dataset_root, 'annotation/test/subset/Resolution.mat')
        self.low_resolution_annotation = loadmat(path)['Test_Size']

    def get_image_path(self, name):
        return os.path.join(self.image_path, name)

    def get_image_annotation(self):
        return self.image_annotation[0]

    def get_train_size(self):
        return self.person_annotation.shape[0]

    def get_test_pool_size(self):
        return self.pool_annotation.shape[0]

    def get_test_image_name(self, i):
        return self.pool_annotation[i,0][0]

    def get_test_query_size(self):
        return self.test_query.shape[1]

    def get_test_query_gallery_size(self):
        return self.test_query[0,0]['Gallery'].shape[1]

    def get_test_query_query_data(self, i):
        # ['imname', 'idlocate', 'ishard', 'idname']
        o = self.test_query[0,i]['Query'][0,0]
        return TestData(o['imname'][0], o['idlocate'][0], o['ishard'][0], o['idname'][0])

    def get_test_query_gallery_data(self, i, j):
        # ['imname', 'idlocate', 'ishard']
        o = self.test_query[0,i]['Gallery'][0,j]
        return TestData(o['imname'][0], o['idlocate'][0], o['ishard'][0], None)
