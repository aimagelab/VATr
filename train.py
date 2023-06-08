import argparse
import random
import shutil
import time
import os

import numpy as np
import torch
import wandb

from data.dataset import TextDataset, TextDatasetval, CollectionTextDataset
from models.model import VATr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--feat_model_path", type=str, default='files/resnet_18_pretrained.pth')
    parser.add_argument("--label_encoder", default='default', type=str)
    parser.add_argument("--save_model_path", default='saved_models', type=str)
    parser.add_argument("--dataset", default='IAM', type=str)
    parser.add_argument("--english_words_path", default='files/english_words.txt', type=str)
    parser.add_argument("--wandb", action='store_true')

    parser.add_argument("--no_writer_loss", action='store_true')
    parser.add_argument("--writer_loss_weight", type=float, default=1.0)
    parser.add_argument("--no_ocr_loss", action='store_true')

    parser.add_argument("--img_height", default=32, type=int)
    parser.add_argument("--resolution", default=16, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_examples", default=15, type=int)

    parser.add_argument("--tn_hidden_dim", default=512, type=int)
    parser.add_argument("--tn_dropout", default=0.1, type=float)
    parser.add_argument("--tn_nheads", default=8, type=int)
    parser.add_argument("--tn_dim_feedforward", default=512, type=int)
    parser.add_argument("--tn_enc_layers", default=3, type=int)
    parser.add_argument("--tn_dec_layers", default=3, type=int)

    parser.add_argument("--alphabet", default='Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%', type=str)
    parser.add_argument("--special_alphabet", default='ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω', type=str)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--g_lr", default=0.00005, type=float)
    parser.add_argument("--d_lr", default=0.00005, type=float)
    parser.add_argument("--w_lr", default=0.00005, type=float)
    parser.add_argument("--ocr_lr", default=0.00005, type=float)
    parser.add_argument("--epochs", default=100_000, type=int)
    parser.add_argument("--num_critic_gocr_train", default=2, type=int)
    parser.add_argument("--num_critic_docr_train", default=1, type=int)
    parser.add_argument("--num_critic_gwl_train", default=2, type=int)
    parser.add_argument("--num_critic_dwl_train", default=1, type=int)
    parser.add_argument("--num_fid_freq", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=742, type=int)
    parser.add_argument("--is_seq", default=True, type=bool)
    parser.add_argument("--num_words", default=3, type=int)
    parser.add_argument("--is_cycle", default=False, type=bool)
    parser.add_argument("--is_kld", default=False, type=bool)
    parser.add_argument("--add_noise", default=False, type=bool)
    parser.add_argument("--all_chars", default=False, type=bool)
    parser.add_argument("--save_model", default=5, type=int)
    parser.add_argument("--save_model_history", default=500, type=int)
    parser.add_argument("--tag", default='debug', type=str)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--query_input", default='unifont', type=str)
    parser.add_argument("--query_linear", default=True, type=bool)

    args = parser.parse_args()
    rSeed(args.seed)
    dataset = CollectionTextDataset(
        args.dataset, 'files', TextDataset, num_examples=args.num_examples,
        collator_resolution=args.resolution, min_virtual_size=283  # IAM 339, CVL 283
    )
    datasetval = CollectionTextDataset(
        args.dataset, 'files', TextDatasetval, num_examples=args.num_examples,
        collator_resolution=args.resolution, min_virtual_size=27  # IAM 161, CVL 27
    )
    args.num_writers = dataset.num_writers

    if args.dataset == 'IAM' or args.dataset == 'CVL':
        args.alphabet = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    else:
        args.alphabet = ''.join(sorted(set(dataset.alphabet + datasetval.alphabet)))
        args.special_alphabet = ''.join(c for c in args.special_alphabet if c not in dataset.alphabet)
    args.vocab_size = len(args.alphabet) if args.vocab_size is None else args.vocab_size
    if not args.is_seq: args.num_words = args.num_examples
    args.exp_name = f"{args.dataset}-{args.num_writers}-{args.num_examples}-E{args.tn_enc_layers}D{args.tn_dec_layers}-LR{args.g_lr}-bs{args.batch_size}-{args.tag}"

    config = {k: v for k, v in args.__dict__.items() if isinstance(v, (bool, int, str, float))}
    args.wandb = args.wandb and torch.cuda.get_device_name(0) != 'Tesla K80'
    wandb_id = wandb.util.generate_id()

    MODEL_PATH = os.path.join(args.save_model_path, args.exp_name)
    os.makedirs(MODEL_PATH, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        collate_fn=dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        datasetval,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        collate_fn=datasetval.collate_fn)

    model = VATr(args)
    start_epoch = 0

    del config['alphabet']
    del config['special_alphabet']

    wandb_params = {
        'project': 'VATr',
        'config': config,
        'name': args.exp_name,
        'id': wandb_id
    }

    checkpoint_path = os.path.join(MODEL_PATH, 'model.pth')

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        wandb_params['id'] = checkpoint['wandb_id']
        wandb_params['resume'] = True
        print(checkpoint_path + ' : Model loaded Successfully')
    elif args.resume:
        raise FileNotFoundError(f'No model found at {checkpoint_path}')
    else:
        if args.feat_model_path is not None and args.feat_model_path.lower() != 'none':
            print('Loading...', args.feat_model_path)
            assert os.path.exists(args.feat_model_path)
            checkpoint = torch.load(args.feat_model_path)
            checkpoint['model']['conv1.weight'] = checkpoint['model']['conv1.weight'].mean(1).unsqueeze(1)
            miss, unexp = model.netG.Feat_Encoder.load_state_dict(checkpoint['model'], strict=False)
            assert unexp == ['fc.weight', 'fc.bias']
            if not os.path.isdir(MODEL_PATH): os.mkdir(MODEL_PATH)
        else:
            print(f'WARNING: No resume of Resnet-18, starting from scratch')

    if args.wandb:
        wandb.init(**wandb_params)
        wandb.watch(model)

    print(f"Starting training")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        log_time = time.time()

        for i, data in enumerate(train_loader):
            if (i % args.num_critic_gocr_train) == 0:
                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % args.num_critic_docr_train) == 0:
                model._set_input(data)
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()

            if (i % args.num_critic_gwl_train) == 0:
                model._set_input(data)
                model.optimize_G_WL()
                model.optimize_G_step()

            if (i % args.num_critic_dwl_train) == 0:
                model._set_input(data)
                model.optimize_D_WL()
                model.optimize_D_WL_step()

            if time.time() - log_time > 10:
                print(f'Epoch {epoch} {i / len(train_loader) * 100:.02f}% running, current time: {time.time() - start_time:.2f} s')
                log_time = time.time()

        end_time = time.time()
        data_val = next(iter(val_loader))
        losses = model.get_current_losses()
        page = model._generate_page(model.sdata, model.input['swids'])
        page_val = model._generate_page(data_val['simg'].to(args.device), data_val['swids'])

        if args.wandb: wandb.log({
            'loss-G': losses['G'],
            'loss-D': losses['D'],
            'loss-Dfake': losses['Dfake'],
            'loss-Dreal': losses['Dreal'],
            'loss-OCR_fake': losses['OCR_fake'],
            'loss-OCR_real': losses['OCR_real'],
            'loss-w_fake': losses['w_fake'],
            'loss-w_real': losses['w_real'],
            'epoch': epoch,
            'timeperepoch': end_time - start_time,
            'result': [wandb.Image(page, caption="page"), wandb.Image(page_val, caption="page_val")]
        })

        print({'EPOCH': epoch, 'TIME': end_time - start_time, 'LOSSES': losses})

        checkpoint = {
            'model': model.state_dict(),
            'wandb_id': wandb_id,
            'epoch': epoch
        }
        if epoch % args.save_model == 0: torch.save(checkpoint, os.path.join(MODEL_PATH, 'model.pth'))
        if epoch % args.save_model_history == 0: torch.save(checkpoint, os.path.join(MODEL_PATH, f'{epoch:04d}_model.pth'))


def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        wandb.finish()
        raise
