import json

def load_json(path):
    with open(path, 'r') as fp:
        json_dict = json.load(fp)
    return json_dict

dt1_path = '/home/kv_zhao/datasets/prelabel/PANDAValidation/results_from_nctu/train/detres.json'
dt2_path = '/home/kv_zhao/datasets/prelabel/PANDAValidation/clustercrowd/vbox_hg_nctu_flip/merged_detres.json'

#dt1_path = '/home/kv_zhao/datasets/prelabel/PANDAValidation/results_from_nctu/test/detres.json'
#dt2_path = '/home/kv_zhao/datasets/prelabel/PANDASubmission/clustercrowd/vbox_hg_nctu/detres.json'

dt1 = load_json(dt1_path)
dt2 = load_json(dt2_path)

print(len(dt1), len(dt2))

def gentable(detres):
    table = {}
    for det in detres:
        img_id = det['image_id']
        if img_id not in table:
            table[img_id] = []
        table[img_id].append(det)
    return table

Dt1 = gentable(dt1)
Dt2 = gentable(dt2)

print(len(Dt1.keys()), len(Dt2.keys()))

for img_id in Dt1:
    pred1 = Dt1[img_id]
    pred2 = Dt2[img_id]

    print(len(pred1), len(pred2))

    confP1 = [p['score'] for p in pred1]
    confP2 = [p['score'] for p in pred2]

    print(confP1)
    print(confP2)

    break