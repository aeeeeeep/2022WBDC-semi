import logging
import os
import torch
from tqdm import tqdm
from lxmert_model_pretrain import LXMERT_PRE
from config import parse_args
from data_helper import create_dataloaders
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate


def validate(model, val_dataloader):
    model.eval()
    mlm_losses, itm_losses = [], []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            mlm_loss, itm_loss, _, _ = model(batch[0], batch[1], batch[2], batch[3])
            mlm_losses.append(mlm_loss.mean().to('cpu').item())
            itm_losses.append(itm_loss.mean().to('cpu').item())
    model.train()
    return sum(mlm_losses)/len(mlm_losses) + sum(itm_losses)/len(itm_losses)


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args, pretrain=True)

    # 2. build model and optimizers
    model = LXMERT_PRE(args)

    num_total_steps = len(train_dataloader) * args.max_epochs

    optimizer, scheduler = build_optimizer(args, model, num_total_steps)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    accumulation_steps = 1
    # loss_min = validate(model, val_dataloader)
    loss_min = 500

    # start_time = time.time()
    for epoch in range(args.max_epochs):

        with tqdm(total=len(train_dataloader)) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            _tqdm.set_description('epoch: {}/{}'.format(epoch, args.max_epochs - 1))  # 设置前缀 一般为epoch的信息
            for batch in train_dataloader:
                model.train()

                mlm_loss, itm_loss, mlm_accuracy, itm_accuracy = model(batch[0], batch[1], batch[2], batch[3])
                mlm_loss = mlm_loss.mean() / accumulation_steps
                itm_loss = itm_loss.mean() / accumulation_steps

                mlm_accuracy = mlm_accuracy.mean()
                itm_accuracy = itm_accuracy.mean()
                # loss = 3 * torch.log(mlm_loss + 1e-12) + 0.2 * torch.log(itm_loss + 1e-12)
                # loss = torch.log(mlm_loss + 1e-12) + torch.log(itm_loss + 1e-12)
                loss = mlm_loss + itm_loss + 1e-12
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                step += 1

                if (step % 3000 == 0):
                    val_loss = validate(model, val_dataloader)
                    logging.info(f"Epoch {epoch} step {step}: loss {val_loss:.3f}")
                    if val_loss < loss_min:
                        loss_min = val_loss
                        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mlm_loss': mlm_loss, 'itm_loss':itm_loss},
                                   f'{args.savedpremodel_path}/pretrain_model_epoch_{epoch}_loss_{loss_min}.bin')

                if step % 10000 == 0:
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save(
                        {'epoch': epoch, 'model_state_dict': state_dict, 'mlm_loss': mlm_loss, 'itm_loss': itm_loss},
                        f'{args.savedpremodel_path}/pretrain_model_pretrain_best_step{step}.bin')

                _tqdm.set_postfix(mlm_loss='{:.3f}'.format(mlm_loss), itm_loss='{:.3f}'.format(itm_loss), mlm_acc='{:.3f}'.format(mlm_accuracy), itm_acc='{:.3f}'.format(itm_accuracy))  # 设置你想要在本次循环内实时监视的变量  可以作为后缀打印出来
                _tqdm.update(1)


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