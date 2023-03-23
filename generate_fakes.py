from models.model import VATr
import argparse
from train import rSeed
import os
import torch
from data.dataset import TextDatasetval, TextDataset, FidDataset
from pathlib import Path

def load_checkpoint(model, checkpoint):
    old_model = model.state_dict()
    if len(checkpoint.keys()) == 241:  # default
        counter = 0
        for k, v in checkpoint.items():
            if k in old_model:
                old_model[k] = v
                counter += 1
            elif 'netG.' + k in old_model:
                old_model['netG.' + k] = v
                counter += 1

        ckeys = [k for k in checkpoint.keys() if 'Feat_Encoder' in k]
        okeys = [k for k in old_model.keys() if 'Feat_Encoder' in k]
        for ck, ok in zip(ckeys, okeys):
            old_model[ok] = checkpoint[ck]
            counter += 1
        assert counter == 241
        checkpoint_dict = old_model
    else:
        assert len(old_model) == len(checkpoint)
        checkpoint_dict = {k2: v1 for (k1, v1), (k2, v2) in zip(checkpoint.items(), old_model.items()) if
                           v1.shape == v2.shape}
    assert len(old_model) == len(checkpoint_dict)
    model.load_state_dict(checkpoint_dict, strict=False)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default='saved_images')
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

    parser.add_argument("--alphabet",
                        default='Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%',
                        type=str)
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

    if args.dataset == 'IAM':
        args.dataset_path = 'files/IAM-32.pickle'
        args.num_writers = 339
    elif args.dataset == 'CVL':
        args.dataset_path = 'files/CVL-32.pickle'
        args.num_writers = 283
    else:
        raise ValueError

    args.vocab_size = len(args.alphabet)
    if not args.is_seq: args.num_words = args.num_examples

    def filter_nums(loader):
        for data in loader:
            numeric_labels = [l.decode('utf-8').isnumeric() for l in data['label']]
            if not any(numeric_labels): continue
            numeric_labels = torch.tensor(numeric_labels).to(args.device)
            data = {
                'img': data['img'][numeric_labels],
                'label': [l for l, b in zip(data['label'], numeric_labels) if b == True],
                'swids': data['swids'][numeric_labels],
                'simg': data['simg'][numeric_labels],
                'wcl': data['wcl'][numeric_labels],
            }
            yield data

    dataset_train = FidDataset(base_path=args.dataset_path, num_examples=args.num_examples, collator_resolution=args.resolution, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
        collate_fn=dataset_train.collate_fn
    )

    dataset_test = FidDataset(base_path=args.dataset_path, num_examples=args.num_examples, collator_resolution=args.resolution, mode='test')
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True, drop_last=False,
        collate_fn=dataset_test.collate_fn
    )

    model = VATr(args)

    args.output = Path(args.output) / Path(args.checkpoint).stem
    print(f'Loading checkpoint {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint)
    epoch = 'unknown'
    if 'epoch' in checkpoint: epoch = checkpoint['epoch']
    if 'model' in checkpoint: checkpoint = checkpoint['model']

    load_checkpoint(model, checkpoint)

    model.eval()
    with torch.no_grad():
        model.save_images_for_fid_calculation(args.output, train_loader, 'train')
        model.save_images_for_fid_calculation(args.output, test_loader, 'test')
    print('Done')


