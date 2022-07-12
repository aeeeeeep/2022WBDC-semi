import logging
import os
import time
import torch
import copy
from albef_model import ALBEF
from config import parse_args
from data_helper import create_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from pgd import PGD
from ema import EMA
from swa import swa


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = ALBEF(args)

    # swa_raw_model = copy.deepcopy(model)

    ema = EMA(model, 0.999, device=args.device)
    ema.register()

    pgd = PGD(model)
    K = 3

    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch['frame_input'],batch['frame_mask'],batch['title_input'],batch['title_mask'],batch['title_token_type_ids'],batch['label'])
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                adv_loss, _, _, _ = model(batch['frame_input'],batch['frame_mask'],batch['title_input'],batch['title_mask'],batch['title_token_type_ids'],batch['label'])
                adv_loss = adv_loss.mean()
                adv_loss.backward()
            pgd.restore()

            optimizer.step()
            ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(
                    f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        ema.apply_shadow()

        # 4. validation
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        ema.restore()
    # swa(swa_raw_model, args.swa_savedmodel_path, swa_start=args.swa_start)


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()