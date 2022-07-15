import torch
from util import evaluate
from albef_model_pretrain import ALBEF_PRE
from config import parse_args
from data_helper import create_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from fgm import FGM
from ema import EMA
import time
import logging
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_pred_and_loss(model, frame_input, frame_mask, text_input, text_mask, label,task=['mlm', 'itm']):
    frame_input.to(DEVICE)
    frame_mask.to(DEVICE)
    text_input.to(DEVICE)
    text_mask.to(DEVICE)

    loss, accuracy, pred_label_id, label = model(frame_input, frame_mask, text_input, text_mask, label)
    return loss, accuracy, pred_label_id, label

def eval(model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            loss, _, pred_label_id, label = get_pred_and_loss(batch['frame_input'], batch['frame_mask'], batch['title_input'],
                                                  batch['title_mask'], batch['label'])
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results

def train(args, get_pred_and_loss):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = ALBEF_PRE(args)

    ema = EMA(model, 0.999, device=args.device)
    ema.register()

    fgm = FGM(model)

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
            loss, accuracy, _, _ = model(batch['frame_input'], batch['frame_mask'], batch['title_input'],
                                         batch['title_mask'], batch['label'])
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            '''fgm'''
            fgm.attack()  # 在embedding上添加对抗扰动
            adv_loss, _, _, _ = model(batch['frame_input'], batch['frame_mask'], batch['title_input'],
                                      batch['title_mask'], batch['label'])
            adv_loss = adv_loss.mean()
            adv_loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数

            optimizer.step()
            ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1

        logging.info(f"Train: Epoch {epoch} : loss {loss:.3f}, "
                     # f"train_acc {train_acc:.3f}")
                     f"adv_loss {adv_loss:.3f}, accuracy {accuracy:.3f}")

        ema.apply_shadow()

        # 4. validation
        loss, results = eval(model, val_dataloader, get_pred_and_loss=get_pred_and_loss)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Validate: Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        ema.restore()


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train(args, get_pred_and_loss=get_pred_and_loss)


if __name__ == '__main__':
    main()