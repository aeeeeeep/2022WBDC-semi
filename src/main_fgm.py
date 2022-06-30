import logging
import os
import glob
import re
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
import warnings
warnings.filterwarnings('ignore')

from config import parse_args
from data_helper import create_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from model import MultiModal
from third_party.fgm import FGM
from third_party.ema import EMA

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating data..."):
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results

def train_and_validate(args, fold_idx):
    fold_idx += 1
    train_dataloader, val_dataloader = create_dataloaders(args, fold_idx)

    savemodel = Path(args.savedmodel_path)
    if savemodel.exists():
        savemodel = savemodel.with_suffix('')
        sep = ''
        dirs = glob.glob(f"{savemodel}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % savemodel.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        savemodel = Path(f"{savemodel}{sep}{n}{savemodel.suffix}")

    os.makedirs(savemodel, exist_ok=True)

    with open(savemodel / f'fold_{fold_idx}.yaml', 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    model = MultiModal(args)

    ema = EMA(model, 0.999, device=args.device)
    ema.register()

    fgm = FGM(model)

    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    step = 0
    best_score = args.best_score
    loss_train, loss_adv, loss_val = 0, 0, 0

    logging.info('-' * 50 + f'[Fold_{fold_idx} Training]' + '-' * 40)

    for epoch in range(args.max_epochs):
        for batch in tqdm(train_dataloader, desc="Training data ..."):
            model.train()
            loss_train, _, _, _ = model(batch)
            loss_train = loss_train.mean()
            loss_train.backward()

            fgm.attack()
            loss_adv, _, _, _ = model(batch)
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
            optimizer.step()
            ema.update()

            optimizer.zero_grad()
            scheduler.step()

            step += 1

        logging.info(f"Train: Epoch {epoch} : train_loss {loss_train:.3f}, "
                     f"adv_loss {loss_adv:.3f}, train_acc {loss_val:.3f}")

        ema.apply_shadow()
        loss_val, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Validate: Epoch {epoch} step {step}: loss_val {loss_val:.3f}, {results}")

        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{savemodel}/epoch_{epoch}_meanf1_{mean_f1}.bin')
        ema.restore()

def main():
    args = parse_args()
    setup_device(args)
    setup_seed(args)

    logging.info("Training/evaluation parameters: %s", args)

    for k in range(args.k_folds):
        train_and_validate(args, k)

if __name__ == '__main__':
    main()
