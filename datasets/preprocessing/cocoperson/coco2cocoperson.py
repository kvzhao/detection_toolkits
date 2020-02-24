
import json
from pycocotools.coco import COCO

"""Note
    {'supercategory': 'person', 'id': 1, 'name': 'person'}
"""

def main(args):
    if args.input_path is None:
        raise ValueError('')
    if args.output_path is None:
        raise ValueError('')

    input_path = args.input_path
    output_path = args.output_path

    coco = COCO(input_path)
    person_category_id = 1

    categories = [
        {
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
        }
    ]

    person_image_ids = coco.getImgIds(catIds=person_category_id)
    print('#of image ids with person: ', len(person_image_ids))
    images = coco.loadImgs(person_image_ids)
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=person_image_ids))
    print('#of annotations with person:', len(annotations))

    person_annotations = [
        anno for anno in annotations if anno['category_id'] == person_category_id]

    json_dict = {
        'images': images,
        'annotations': person_annotations,
        'categories': categories,
    }
    with open(output_path, 'w') as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    print('Done, save to {}'.format(output_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default=None, help='')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='')
    args = parser.parse_args()
    main(args)
