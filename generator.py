from models.model import VATr
import argparse
import torch
import collections
from data.dataset import FolderDataset
import numpy as np
import cv2

def load_checkpoint(model, checkpoint):
    if not isinstance(checkpoint, collections.OrderedDict):
        checkpoint = checkpoint['model']
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

class FakeArgs:
    feat_model_path = 'files/resnet_18_pretrained.pth'
    label_encoder = 'default'
    save_model_path = 'saved_models'
    dataset = 'IAM'
    english_words_path = 'files/english_words.txt'
    wandb = False
    no_writer_loss = False
    writer_loss_weight = 1.0
    no_ocr_loss = False
    img_height = 32
    resolution = 16
    batch_size = 32
    num_workers = 4
    num_epochs = 100
    lr = 0.0001
    num_examples = 15
    is_seq = True
    is_kld = False
    tn_hidden_dim = 512
    tn_nheads = 8
    tn_dim_feedforward = 512
    tn_dropout = 0.1
    tn_enc_layers = 3
    tn_dec_layers = 3
    alphabet = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
    special_alphabet = 'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'
    query_input = 'unifont'
    query_linear = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(alphabet)
    num_writers = 339  # 339 for IAM, 283 for CVL
    g_lr = 0.00005
    d_lr = 0.00005
    w_lr = 0.00005
    ocr_lr = 0.00005
    add_noise = True
    all_chars = False

class VATr_writer:
    def __init__(self, checkpoint_path, args=FakeArgs()):
        self.model = VATr(args)
        checkpoint = torch.load(checkpoint_path)
        load_checkpoint(self.model, checkpoint)
        self.model.eval()
        self.style_dataset = None

    def set_style_folder(self, style_folder, num_examples=15):
        self.style_dataset = FolderDataset(style_folder, num_examples=num_examples)

    @torch.no_grad()
    def generate(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.style_dataset is None:
            raise Exception('Style is not set')

        gap = np.ones((32, 16))
        fakes = []
        for i, text in enumerate(texts, 1):
            print(f'[{i}/{len(texts)}] Generating for text: {text}')
            style = self.style_dataset.sample_style()
            style_imgs = style['simg'].unsqueeze(0).to(self.model.args.device)

            text_encode, len_text, encode_pos = self.model.netconverter.encode(text.split())
            text_encode = text_encode.to(self.model.args.device).unsqueeze(0)

            fake = self.model._generate_fakes(style_imgs, text_encode, len_text, encode_pos)
            fake = np.concatenate(sum([[img, gap] for img in fake], []), axis=1)[:, :-16]
            fake = (fake * 255).astype(np.uint8)
            fakes.append(fake)
        return fakes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--style-folder", default='files/style_samples/00', type=str)
    parser.add_argument("-t", "--text", default='Tha\'s one small step for man, one giant leap for mankind ΑαΒβΓγΔδ', type=str)
    parser.add_argument("-c", "--checkpoint", default='files/vatr.pth', type=str)
    parser.add_argument("-o", "--output", default='files/output.png', type=str)
    args = parser.parse_args()

    writer = VATr_writer(args.checkpoint)
    writer.set_style_folder(args.style_folder)
    fakes = writer.generate(args.text)
    assert len(fakes) == 1
    cv2.imwrite(args.output, fakes[0])
    print('Done')
