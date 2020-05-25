# NightOwls API

This is the SDK for NightOwls dataset (http://www.nightowls-dataset.org/)

We provide two APIs:
## 1) Python
The data format/API is fully compatible with MS-COCO (http://cocodataset.org/)

`load_training_data_demo.py` - script demonstrating loading the dataset

`eval.py` - pedestrian detection evaluation

`sample-Faster-RCNN-nightowls_validation.json` - sample Faster RCNN pedestrian detector output on the validation set. Achieves average Miss Rate (MR) of 21.54% @ Reasonable setting



## 2) Caltech Pedestrians format (Matlab)
The data format is compatible with Caltech Pedestrians dataset (http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

`demo.m` - script demonstrating loading the dataset

`matDetectionsToJson.m` - convert detections in .mat format into a JSON format (MSCOCO compatible)

`txtDetectionsToJson.m` - convert detections in .txt format into a JSON format (MSCOCO compatible)

Note that we provide the evaluation code only in Python (above).

## License

This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we (University of Oxford) do not accept any responsibility for errors or omissions.
2. That you include a reference to the Nightowls Dataset in any work that makes use of the dataset.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. You may not use the dataset or any derivative work for commercial purposes such as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (University of Oxford).


## Acknowledgement

When using the dataset, please cite


```
@inproceedings{Nightowls,
  title={NightOwls: A pedestrians at night dataset},
  author={Neumann, Luk{\'a}{\v{s}} and Karg, Michelle and Zhang, Shanshan and Scharfenberger, Christian and Piegert, Eric and Mistr, Sarah and Prokofyeva, Olga and Thiel, Robert and Vedaldi, Andrea and Zisserman, Andrew and Schiele, Bernt},
  booktitle={Asian Conference on Computer Vision},
  pages={691--705},
  year={2018},
  organization={Springer}
}
```
