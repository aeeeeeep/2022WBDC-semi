import os
import torch
import time
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from tqdm import tqdm
from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal

def inference():
    args = parse_args()
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    path = "weights/"
    dirs = os.listdir(path)
    weights_file = []
    for dir in dirs:
        dir = os.path.join(os.path.abspath(path), dir)
        if os.path.isdir(dir):
            weights_dir = os.listdir(dir)
            weight_dict = {}
            for weight in weights_dir:
                weight = os.path.join(dir, weight)
                weight_time = time.localtime(os.stat(weight).st_mtime)
                weight_dict[weight_time] = weight
            weights_file.append(weight_dict[max(weight_dict.keys())])

    pred_matrix = np.zeros((args.num_weights, len(dataset), 200))

    for k in range(args.num_weights):
        model = MultiModal(args)
        checkpoint = torch.load(weights_file[k], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()

        # 3. inference
        temp = np.zeros((1, 200))
        with torch.no_grad():
            for batch in tqdm(dataloader):
                prediction = model(batch, inference=True)
                temp = np.vstack((temp, prediction.cpu().numpy()))
        pred_matrix[k] = temp[1:, :]

    predict_mean = np.zeros((len(dataset), 200))

    for m in range(predict_mean.shape[0]):
        for n in range(predict_mean.shape[1]):
            for k in range(args.num_weights):
                predict_mean[m][n] += pred_matrix[k][m][n]
            predict_mean[m][n] /= args.num_weights

    predictions = []
    for row in range(predict_mean.shape[0]):
        predictions.append(np.argmax(predict_mean[row]))

    with open(args.test_output_csv, 'w') as f:
        for predict_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(predict_label_id)
            f.write(f'{video_id},{category_id}\n')

if __name__ == '__main__':
    inference()
