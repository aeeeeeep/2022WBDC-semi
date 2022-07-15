import logging
import os
import torch
from tqdm import tqdm
from albef_model_pretrain import ALBEF_PRE
from config import parse_args
from data_helper_pretrain import create_pretrain_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer

def validate(model, val_dataloader):
    model.eval()
    loss_mlm, loss_ita, loss_itm = [], [], []
    with torch.no_grad():
        for batch in val_dataloader:
            loss_mlm, loss_ita, loss_itm = model(batch['frame_input'],batch['frame_mask'],batch['title_input'],batch['title_mask'])
            loss_mlm.append(loss_mlm.mean().to('cpu').item())
            loss_ita.append(loss_ita.mean().to('cpu').item())
            loss_itm.append(loss_itm.mean().to('cpu').item())

    model.train()
    return sum(loss_mlm)/len(loss_mlm) + sum(loss_ita)/len(loss_ita) + 0.1 * sum(loss_itm)/len(loss_itm)



def pretrain(args):
    # 1. load data
    train_dataloader, val_dataloader = create_pretrain_dataloaders(args)

    # 2. build model and optimizers
    model = ALBEF_PRE(args)


    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    loss_min = validate(model, val_dataloader)
    accumulation_steps = 1
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        with tqdm(total=len(train_dataloader)) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            _tqdm.set_description('epoch: {}/{}'.format(epoch, args.max_epochs - 1))  # 设置前缀 一般为epoch的信息
            for batch in train_dataloader:
                model.train()

                loss_mlm, loss_ita, loss_itm = model(batch['frame_input'],batch['frame_mask'],batch['title_input'],batch['title_mask'])
                loss_mlm = loss_mlm.mean() / accumulation_steps
                loss_ita = loss_ita.mean() / accumulation_steps
                loss_itm = loss_itm.mean() / accumulation_steps
                loss = 3 * torch.log(loss_mlm + 1e-12) + 0.2 * torch.log(loss_ita + 1e-12) + torch.log(loss_itm + 1e-12)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                step += 1

                if step % 3000 == 0:
                    # 5. save checkpoint
                    val_loss = validate(model, val_dataloader)
                    if val_loss < loss_min:
                        loss_min = val_loss
                        logging.info(f"Saveing model, val loss: {val_loss}")
                        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                        torch.save(
                            {'step': step, 'model_state_dict': state_dict, 'loss_mlm': loss_mlm, 'loss_ita': loss_ita, 'loss_itm': loss_itm},
                            f'{args.savedmodel_path}/model_pretrain_best.bin')

                if step % 10000 == 0:
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save(
                        {'epoch': epoch, 'model_state_dict': state_dict, 'loss_mlm': loss_mlm, 'loss_ita': loss_ita, 'loss_itm': loss_itm},
                        f'{args.savedpremodel_path}/model_pretrain_best_step{step}.bin')
                _tqdm.set_postfix(loss='{:.3f}'.format(loss))  # 设置你想要在本次循环内实时监视的变量  可以作为后缀打印出来
                _tqdm.update(1)  # 设置你每一次想让进度条更新的iteration 大小

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    pretrain(args)


if __name__ == '__main__':
    main()
