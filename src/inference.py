import torch
import tqdm
from torch.utils.data import SequentialSampler, DataLoader

from prefetch_generator import BackgroundGenerator
from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from lxmert_model import LXMERT


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoaderX(dataset,
                             batch_size=args.test_batch_size,
                             sampler=sampler,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=args.num_workers,
                             prefetch_factor=args.prefetch)

    # 2. load model
    model = LXMERT(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
            pred_label_id = model(batch[0], batch[1], batch[2], batch[3], inference=True)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
