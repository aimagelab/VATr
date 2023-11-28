import torch
from torch.nn import CTCLoss, MSELoss, L1Loss
from torch.nn.utils import clip_grad_norm_
import time
import sys
import random
import torchvision.models as models
from models.transformer import *
from .BigGAN_networks import *
from .OCR_network import *
from models.blocks import Conv2dBlock, ResBlocks
from util.util import toggle_grad, loss_hinge_dis, loss_hinge_gen, ortho, default_ortho, toggle_grad, prepare_z_y, \
    make_one_hot, to_device, multiple_replace, random_word
from models.inception import InceptionV3
import cv2
from .positional_encodings import PositionalEncoding1D
from models.unifont_module import UnifontModule
from PIL import Image
from datetime import timedelta


def get_rgb(x):
    R = 255 - int(int(x > 0.5) * 255 * (x - 0.5) / 0.5)
    G = 0
    B = 255 + int(int(x < 0.5) * 255 * (x - 0.5) / 0.5)
    return R, G, B


def get_page_from_words(word_lists, MAX_IMG_WIDTH=800):
    line_all = []
    line_t = []

    width_t = 0

    for i in word_lists:

        width_t = width_t + i.shape[1] + 16

        if width_t > MAX_IMG_WIDTH:
            line_all.append(np.concatenate(line_t, 1))

            line_t = []

            width_t = i.shape[1] + 16

        line_t.append(i)
        line_t.append(np.ones((i.shape[0], 16)))

    if len(line_all) == 0:
        line_all.append(np.concatenate(line_t, 1))

    max_lin_widths = MAX_IMG_WIDTH  # max([i.shape[1] for i in line_all])
    gap_h = np.ones([16, max_lin_widths])

    page_ = []

    for l in line_all:
        pad_ = np.ones([l.shape[0], max_lin_widths - l.shape[1]])

        page_.append(np.concatenate([l, pad_], 1))
        page_.append(gap_h)

    page = np.concatenate(page_, 0)

    return page * 255


class FCNDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(FCNDecoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        y = self.model(x)

        return y


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        INP_CHANNEL = self.args.num_examples
        if self.args.is_seq:
            INP_CHANNEL = 1

        encoder_layer = TransformerEncoderLayer(self.args.tn_hidden_dim, self.args.tn_nheads,
                                                self.args.tn_dim_feedforward,
                                                self.args.tn_dropout, "relu", True)
        encoder_norm = nn.LayerNorm(self.args.tn_hidden_dim) if True else None
        self.encoder = TransformerEncoder(encoder_layer, self.args.tn_enc_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(self.args.tn_hidden_dim, self.args.tn_nheads,
                                                self.args.tn_dim_feedforward,
                                                self.args.tn_dropout, "relu", True)
        decoder_norm = nn.LayerNorm(self.args.tn_hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, self.args.tn_dec_layers, decoder_norm,
                                          return_intermediate=True)

        # self.Feat_Encoder = nn.Sequential(
        #     *(
        #             [nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(pretrained=True).children())[1:-2]
        #     )
        # )

        self.Feat_Encoder = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.Feat_Encoder.conv1 = nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.Feat_Encoder.fc = nn.Identity()
        self.Feat_Encoder.avgpool = nn.Identity()

        # self.query_embed = nn.Embedding(self.args.vocab_size, self.args.tn_hidden_dim)
        self.query_embed = UnifontModule(
            self.args.tn_dim_feedforward,
            self.args.alphabet + self.args.special_alphabet,
            input_type=self.args.query_input,
            linear=self.args.query_linear
        )
        # self.query_embed = LearnableModule(self.args.tn_dim_feedforward)
        self.pos_encoder = PositionalEncoding1D(self.args.tn_hidden_dim)

        self.linear_q = nn.Linear(self.args.tn_dim_feedforward, self.args.tn_dim_feedforward * 8)

        self.DEC = FCNDecoder(res_norm='in')

        self._muE = nn.Linear(512, 512)
        self._logvarE = nn.Linear(512, 512)

        self._muD = nn.Linear(512, 512)
        self._logvarD = nn.Linear(512, 512)

        self.l1loss = nn.L1Loss()

        self.noise = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))

    def reparameterize(self, mu, logvar):

        mu = torch.unbind(mu, 1)
        logvar = torch.unbind(logvar, 1)

        outs = []

        for m, l in zip(mu, logvar):
            sigma = torch.exp(l)
            eps = torch.cuda.FloatTensor(l.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())

            out = m + sigma * eps

            outs.append(out)

        return torch.stack(outs, 1)

    def reverse_forward(self, ST, QR):
        # Attention Visualization Init

        enc_attn_weights, dec_attn_weights = [], []

        self.hooks = [

            self.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        # Attention Visualization Init

        B, N, R, C = ST.shape
        FEAT_ST = self.Feat_Encoder(ST.view(B * N, 1, R, C))
        FEAT_ST = FEAT_ST.view(B, 512, 1, -1)

        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)

        memory = self.encoder(FEAT_ST_ENC)

        # QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)
        QR_EMB = self.query_embed(QR).permute(1, 0, 2)

        tgt = torch.zeros_like(QR_EMB)
        hs = self.decoder(tgt, memory, query_pos=QR_EMB)
        # hs = self.decoder(QR_EMB, memory, query_pos=query_pos)

        h = hs.transpose(1, 2)[-1]  # torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)

        if self.args.add_noise:
            h = h + self.noise.sample(h.size()).squeeze(-1).to(self.args.device)

        h = self.linear_q(h)
        h = h.contiguous()

        h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
        h = h.permute(0, 3, 2, 1)

        h = self.DEC(h)

        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()

        for hook in self.hooks:
            hook.remove()

        return h

    def Eval(self, ST, QRS, QRS_pos):

        if self.args.is_seq:
            B, N, R, C = ST.shape
            FEAT_ST = self.Feat_Encoder(ST.view(B * N, 1, R, C))
            FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
        else:
            FEAT_ST = self.Feat_Encoder(ST)

        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)

        memory = self.encoder(FEAT_ST_ENC)

        if self.args.is_kld:
            Ex = memory.permute(1, 0, 2)

            memory_mu = self._muE(Ex)
            memory_logvar = self._logvarE(Ex)

            memory = self.reparameterize(memory_mu, memory_logvar).permute(1, 0, 2)

        OUT_IMGS = []

        for i in range(QRS.shape[1]):

            QR = QRS[:, i, :]
            if QRS_pos is not None:
                QR_pos = [QRS_pos[i] for _ in range(QRS.shape[0])]

            # if self.args.all_chars:
            #     QR_EMB = self.query_embed.weight.repeat(self.args.self.args.batch_size, 1, 1).permute(1, 0, 2)
            # else:
            #     QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)
            QR_EMB = self.query_embed(QR).permute(1, 0, 2)

            tgt = torch.zeros_like(QR_EMB)
            # query_pos = self.pos_encoder(QR_EMB)

            # hs = self.decoder(tgt, memory, query_pos=QR_EMB)
            hs = self.decoder(tgt, memory, query_pos=QR_EMB)

            if self.args.is_kld:
                Dx = hs[0].permute(1, 0, 2)

                hs_mu = self._muD(Dx)
                hs_logvar = self._logvarD(Dx)

                hs = self.reparameterize(hs_mu, hs_logvar).permute(1, 0, 2).unsqueeze(0)

            h = hs.transpose(1, 2)[-1]  # torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)
            if self.args.add_noise:
                h = h + self.noise.sample(h.size()).squeeze(-1).to(self.args.device)

            h = self.linear_q(h)
            h = h.contiguous()

            if self.args.all_chars:
                h = torch.stack([h[i][QR[i]] for i in range(self.args.self.args.batch_size)], 0)

            h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
            h = h.permute(0, 3, 2, 1)

            h = self.DEC(h)

            OUT_IMGS.append(h.detach())

        return OUT_IMGS

    def compute_style(self, ST):
        B, N, R, C = ST.shape
        FEAT_ST = self.Feat_Encoder(ST.view(B * N, 1, R, C))
        FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)
        memory = self.encoder(FEAT_ST_ENC)
        return memory

    def forward(self, ST, QR, QRs=None, QR_pos=None, QRs_pos=None, mode='train'):
        # Attention Visualization Init

        enc_attn_weights, dec_attn_weights = [], []

        self.hooks = [

            self.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        # Attention Visualization Init

        memory = self.compute_style(ST)

        # QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)
        QR_EMB = self.query_embed(QR).permute(1, 0, 2)

        # query_pos = self.pos_encoder(QR_EMB)

        tgt = torch.zeros_like(QR_EMB)
        hs = self.decoder(tgt, memory, query_pos=QR_EMB)
        # hs = self.decoder(QR_EMB, memory, query_pos=query_pos)

        h = hs.transpose(1, 2)[-1]  # torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)

        if self.args.add_noise:
            h = h + self.noise.sample(h.size()).squeeze(-1).to(self.args.device)

        h = self.linear_q(h)
        h = h.contiguous()

        h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
        h = h.permute(0, 3, 2, 1)

        h = self.DEC(h)

        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()

        for hook in self.hooks:
            hook.remove()

        return h, memory


class VATr(nn.Module):

    def __init__(self, args):
        super(VATr, self).__init__()
        self.args = args

        self.epsilon = 1e-7
        self.netG = Generator(self.args).to(self.args.device)
        self.netD = Discriminator(
            resolution=self.args.resolution, n_classes=self.args.vocab_size
        ).to(self.args.device)
        self.netW = WDiscriminator(
            resolution=self.args.resolution, n_classes=self.args.vocab_size, output_dim=self.args.num_writers
        ).to(self.args.device)

        self.netconverter = strLabelConverter(self.args.alphabet + self.args.special_alphabet)

        self.netOCR = CRNN(self.args).to(self.args.device)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction='none')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(self.args.device)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=self.args.g_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizer_OCR = torch.optim.Adam(self.netOCR.parameters(),
                                              lr=self.args.ocr_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=self.args.d_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizer_wl = torch.optim.Adam(self.netW.parameters(),
                                             lr=self.args.w_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizers = [self.optimizer_G, self.optimizer_OCR, self.optimizer_D, self.optimizer_wl]

        self.optimizer_G.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.loss_G = 0
        self.loss_D = 0
        self.loss_Dfake = 0
        self.loss_Dreal = 0
        self.loss_OCR_fake = 0
        self.loss_OCR_real = 0
        self.loss_w_fake = 0
        self.loss_w_real = 0
        self.Lcycle = 0
        self.lda1 = 0
        self.lda2 = 0
        self.KLD = 0

        with open(self.args.english_words_path, 'r') as f:
            self.lex = f.read().splitlines()
        self.lex = [l for l in self.lex if len(l) < 20 and set(l) < set(self.args.alphabet)]

        with open('mytext.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\n', ' ')
            self.text = ''.join(c for c in self.text if c in (self.args.alphabet + self.args.special_alphabet))  # just to avoid problems with the font dataset
            self.text = [word.encode() for word in self.text.split()]  # [:args.num_examples]

        self.eval_text_encode, self.eval_len_text, self.eval_encode_pos = self.netconverter.encode(self.text)
        self.eval_text_encode = self.eval_text_encode.to(self.args.device).repeat(self.args.batch_size, 1, 1)

    def save_images_for_fid_calculation(self, path, loader, split='train'):
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        self.real_base = path / f'Real_{split}'
        self.fake_base = path / f'Fake_{split}'

        self.real_base.mkdir(exist_ok=True)
        self.fake_base.mkdir(exist_ok=True)

        print('Saving images...')

        print('  Saving images on {}'.format(str(self.real_base)))
        print('  Saving images on {}'.format(str(self.fake_base)))
        counter = 0
        ann = {}
        start_time = time.time()
        for step, data in enumerate(loader):
            ST = data['simg'].to(self.args.device)
            texts = [l.decode('utf-8') for l in data['label']]
            # texts = [''.join([c for c in t if c in string.ascii_letters]) for t in texts]
            texts = [t.encode('utf-8') for t in texts]
            self.eval_text_encode, self.eval_len_text, self.eval_encode_pos = self.netconverter.encode(texts)
            # self.eval_text_encode = self.eval_text_encode.to(self.args.device).repeat(self.args.batch_size, 1, 1)
            self.eval_text_encode = self.eval_text_encode.to(self.args.device).unsqueeze(1)
            self.fakes = self.netG.Eval(ST, self.eval_text_encode, self.eval_encode_pos)
            fake_images = torch.cat(self.fakes, 1).detach().cpu().numpy()
            real_images = data['img'].detach().cpu().numpy()
            writer_ids = data['wcl'].int().tolist()

            for i, (fake, real, wid, lb, img_id) in enumerate(zip(fake_images, real_images, writer_ids, data['label'], data['idx'])):
                lb = lb.decode()
                ann[step * self.args.batch_size + i] = lb
                img_id = f'{img_id:05d}.png'
                real_img_path = self.real_base / f"{wid:03d}" / img_id
                fake_img_path = self.fake_base / f"{wid:03d}" / img_id
                real_img_path.parent.mkdir(exist_ok=True, parents=True)
                fake_img_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(real_img_path), 255 * real.squeeze())
                cv2.imwrite(str(fake_img_path), 255 * fake.squeeze())
                counter += 1
            eta = (time.time() - start_time) / (step + 1) * (len(loader) - step - 1)
            eta = str(timedelta(seconds=eta))
            if step % 100 == 0:
                print(f'[{(step + 1) / len(loader) * 100:.02f}%][{counter:05d}] image saved: {real_img_path.name} ETA {eta}')

            with open(path / 'ann.json', 'w') as f:
                json.dump(ann, f)
        return

    def _generate_fakes(self, ST, eval_text_encode=None, eval_len_text=None, eval_encode_pos=None):
        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text
        if eval_encode_pos is None:
            eval_encode_pos = self.eval_encode_pos

        self.fakes = self.netG.Eval(ST, eval_text_encode, eval_encode_pos)

        np_fakes = []
        for batch_idx in range(self.fakes[0].shape[0]):
            for idx, fake in enumerate(self.fakes):
                fake = fake[batch_idx, 0, :, :eval_len_text[idx] * self.args.resolution]
                fake = (fake + 1) / 2
                np_fakes.append(fake.cpu().numpy())
        return np_fakes

    def _generate_page(self, ST, SLEN, eval_text_encode=None, eval_len_text=None, eval_encode_pos=None, lwidth=260, rwidth=980):
        # ST -> Style?

        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text
        if eval_encode_pos is None:
            eval_encode_pos = self.eval_encode_pos

        text_encode, text_len, _ = self.netconverter.encode(self.args.special_alphabet)
        symbols = self.netG.query_embed.symbols[text_encode].reshape(-1, 16, 16).cpu().numpy()
        imgs = [Image.fromarray(s).resize((32, 32), resample=0) for s in symbols]
        special_examples = 1 - np.concatenate([np.array(i) for i in imgs], axis=-1)

        self.fakes = self.netG.Eval(ST, eval_text_encode, eval_encode_pos)

        page1s = []
        page2s = []

        for batch_idx in range(ST.shape[0]):

            word_t = []
            word_l = []

            gap = np.ones([self.args.img_height, 16])

            line_wids = []

            for idx, fake_ in enumerate(self.fakes):

                word_t.append((fake_[batch_idx, 0, :, :eval_len_text[idx] * self.args.resolution].cpu().numpy() + 1) / 2)

                word_t.append(gap)

                if sum(t.shape[-1] for t in word_t) >= rwidth or idx == len(self.fakes) - 1 or (len(self.fakes) - len(self.args.special_alphabet) - 1) == idx:
                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            # add the examples from the UnifontModules
            word_l.append(special_examples)
            line_wids.append(special_examples.shape[1])

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:
                pad_ = np.ones([self.args.img_height, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page1 = np.concatenate(page_, 0)

            word_t = []
            word_l = []

            gap = np.ones([self.args.img_height, 16])

            line_wids = []

            sdata_ = [i.unsqueeze(1) for i in torch.unbind(ST, 1)]

            for idx, st in enumerate((sdata_)):

                word_t.append((st[batch_idx, 0, :, :int(SLEN.cpu().numpy()[batch_idx][idx])].cpu().numpy() + 1) / 2)
                # word_t.append((st[batch_idx, 0, :, :].cpu().numpy() + 1) / 2)

                word_t.append(gap)

                if sum(t.shape[-1] for t in word_t) >= lwidth or idx == len(sdata_) - 1:
                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:
                pad_ = np.ones([self.args.img_height, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page2 = np.concatenate(page_, 0)

            merge_w_size = max(page1.shape[0], page2.shape[0])

            if page1.shape[0] != merge_w_size:
                page1 = np.concatenate([page1, np.ones([merge_w_size - page1.shape[0], page1.shape[1]])], 0)

            if page2.shape[0] != merge_w_size:
                page2 = np.concatenate([page2, np.ones([merge_w_size - page2.shape[0], page2.shape[1]])], 0)

            page1s.append(page1)
            page2s.append(page2)

            # page = np.concatenate([page2, page1], 1)

        page1s_ = np.concatenate(page1s, 0)
        max_wid = max([i.shape[1] for i in page2s])
        padded_page2s = []

        for para in page2s:
            padded_page2s.append(np.concatenate([para, np.ones([para.shape[0], max_wid - para.shape[1]])], 1))

        padded_page2s_ = np.concatenate(padded_page2s, 0)

        # fakes = self.fakes[:15]
        # FEAT1 = self.inception(torch.cat(fakes, 0).repeat(1, 3, 1, 1))
        # FEAT1 = FEAT1[0].detach().view(self.args.batch_size, len(fakes), -1).cpu().numpy()
        #
        # FEAT2 = self.inception(self.sdata.view(self.args.batch_size * self.args.num_examples, 1, 32, -1).repeat(1, 3, 1, 1))
        # FEAT2 = FEAT2[0].detach().view(self.args.batch_size, -1, 2048).cpu().numpy()
        # muvars1 = [{'mu': np.mean(FEAT1[i], axis=0), 'sigma': np.cov(FEAT1[i], rowvar=False)} for i in
        #            range(FEAT1.shape[0])]
        # muvars2 = [{'mu': np.mean(FEAT2[i], axis=0), 'sigma': np.cov(FEAT2[i], rowvar=False)} for i in
        #            range(FEAT2.shape[0])]
        #
        # fid = calculate_frechet_distance(
        #     [muvars['mu'] for muvars in muvars1],
        #     [muvars['sigma'] for muvars in muvars1],
        #     [muvars['mu'] for muvars in muvars2],
        #     [muvars['sigma'] for muvars in muvars2]
        # )

        return np.concatenate([padded_page2s_, page1s_], 1)

    def get_current_losses(self):

        losses = {}

        losses['G'] = self.loss_G
        losses['D'] = self.loss_D
        losses['Dfake'] = self.loss_Dfake
        losses['Dreal'] = self.loss_Dreal
        losses['OCR_fake'] = self.loss_OCR_fake
        losses['OCR_real'] = self.loss_OCR_real
        losses['w_fake'] = self.loss_w_fake
        losses['w_real'] = self.loss_w_real
        losses['cycle'] = self.Lcycle
        losses['lda1'] = self.lda1
        losses['lda2'] = self.lda2
        losses['KLD'] = self.KLD

        return losses

    def load_networks(self, epoch):
        BaseModel.load_networks(self, epoch)
        if self.opt.single_writer:
            load_filename = '%s_z.pkl' % epoch
            load_path = os.path.join(self.save_dir, load_filename)
            self.z = torch.load(load_path)

    def _set_input(self, input):
        self.input = input

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.real = self.input['img'].to(self.args.device)
        self.label = self.input['label']
        self.sdata = self.input['simg'].to(self.args.device)
        self.ST_LEN = self.input['swids']
        self.text_encode, self.len_text, self.encode_pos = self.netconverter.encode(self.label)
        self.one_hot_real = make_one_hot(self.text_encode, self.len_text, self.args.vocab_size).to(
            self.args.device).detach()
        self.text_encode = self.text_encode.to(self.args.device).detach()
        self.len_text = self.len_text.detach()

        self.words = [word.encode('utf-8') for word in random.choices(self.lex, k=self.args.batch_size)]
        self.text_encode_fake, self.len_text_fake, self.encode_pos_fake = self.netconverter.encode(self.words)
        self.text_encode_fake = self.text_encode_fake.to(self.args.device)
        self.one_hot_fake = make_one_hot(self.text_encode_fake, self.len_text_fake, self.args.vocab_size).to(
            self.args.device)

        self.text_encode_fake_js = []
        self.encode_pos_fake_js = []

        for _ in range(self.args.num_words - 1):
            self.words_j = [word.encode('utf-8') for word in random.choices(self.lex, k=self.args.batch_size)]
            self.text_encode_fake_j, self.len_text_fake_j, self.encode_pos_fake_j = self.netconverter.encode(self.words_j)
            self.text_encode_fake_j = self.text_encode_fake_j.to(self.args.device)
            self.text_encode_fake_js.append(self.text_encode_fake_j)
            self.encode_pos_fake_js.append(self.encode_pos_fake_j)

        self.fake, self.style = self.netG(self.sdata, self.text_encode_fake, self.text_encode_fake_js, self.encode_pos_fake, self.encode_pos_fake_js)

    def backward_D_OCR(self):
        self.real.__repr__()
        self.fake.__repr__()
        pred_real = self.netD(self.real.detach())
        pred_fake = self.netD(**{'x': self.fake.detach()})

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(),
                                                          self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake

        if not self.args.no_ocr_loss:
            self.pred_real_OCR = self.netOCR(self.real.detach())
            preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.args.batch_size).detach()
            loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size,
                                               self.len_text.detach())
            self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

            loss_total = self.loss_D + self.loss_OCR_real
        else:
            loss_total = self.loss_D

        # backward
        loss_total.backward()
        if not self.args.no_ocr_loss:
            for param in self.netOCR.parameters():
                param.grad[param.grad != param.grad] = 0
                param.grad[torch.isnan(param.grad)] = 0
                param.grad[torch.isinf(param.grad)] = 0

        return loss_total

    def backward_D_WL(self):
        # Real
        pred_real = self.netD(self.real.detach())

        pred_fake = self.netD(**{'x': self.fake.detach()})

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(),
                                                          self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake

        if not self.args.no_writer_loss:
            self.loss_w_real = self.netW(self.real.detach(), self.input['wcl'].to(self.args.device)).mean()
            # total loss
            loss_total = self.loss_D + self.loss_w_real * self.args.writer_loss_weight
        else:
            loss_total = self.loss_D

        # backward
        loss_total.backward()

        return loss_total

    def optimize_D_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], True)

        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.backward_D_WL()

    def backward_D_OCR_WL(self):
        # Real
        if self.real_z_mean is None:
            pred_real = self.netD(self.real.detach())
        else:
            pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
        # Fake
        try:
            pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        except:
            print('a')
        # Combined loss
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(),
                                                          self.len_text.detach(), self.opt.mask_loss)

        self.loss_D = self.loss_Dreal + self.loss_Dfake
        # OCR loss on real data
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.self.args.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size,
                                           self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])
        # total loss
        self.loss_w_real = self.netW(self.real.detach(), self.wcl)
        loss_total = self.loss_D + self.loss_OCR_real + self.loss_w_real

        # backward
        loss_total.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad != param.grad] = 0
            param.grad[torch.isnan(param.grad)] = 0
            param.grad[torch.isinf(param.grad)] = 0

        return loss_total

    def optimize_D_WL_step(self):
        self.optimizer_D.step()
        self.optimizer_wl.step()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

    def backward_OCR(self):
        # OCR loss on real data
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.self.args.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size,
                                           self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        # backward
        self.loss_OCR_real.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad != param.grad] = 0
            param.grad[torch.isnan(param.grad)] = 0
            param.grad[torch.isinf(param.grad)] = 0

        return self.loss_OCR_real

    def backward_D(self):
        # Real
        if self.real_z_mean is None:
            pred_real = self.netD(self.real.detach())
        else:
            pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
        pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        # Combined loss
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(),
                                                          self.len_text.detach(), self.opt.mask_loss)
        self.loss_D = self.loss_Dreal + self.loss_Dfake
        # backward
        self.loss_D.backward()

        return self.loss_D

    def compute_cycle_loss(self):
        fake_input = torch.ones_like(self.sdata)
        width = min(self.sdata.size(-1), self.fake.size(-1))
        fake_input[:, :, :, :width] = self.fake.repeat(1, 15, 1, 1)[:, :, :, :width]
        with torch.no_grad():
            fake_style = self.netG.compute_style(fake_input)
        return torch.sum(torch.abs(self.style.detach() - fake_style), dim=1).mean()

    def backward_G_only(self):

        self.gb_alpha = 0.7
        if self.args.is_cycle:
            self.Lcycle = self.compute_cycle_loss()

        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()

        if not self.args.no_ocr_loss:
            pred_fake_OCR = self.netOCR(self.fake)
            preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.args.batch_size).detach()
            loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size,
                                               self.len_text_fake.detach())
            self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

        self.loss_G = self.loss_G + self.Lcycle + self.lda1 + self.lda2 - self.KLD

        if not self.args.no_ocr_loss:
            self.loss_T = self.loss_G + self.loss_OCR_fake
        else:
            self.loss_T = self.loss_G

        if not self.args.no_ocr_loss:
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)

        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        self.loss_T.backward(retain_graph=True)

        if not self.args.no_ocr_loss:
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR))
            self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
            self.loss_T = self.loss_G + self.loss_OCR_fake
        else:
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = 1
            self.loss_T = self.loss_G

        if a is None:
            print(self.loss_OCR_fake, self.loss_G, torch.std(grad_fake_adv))
        if a > 1000 or a < 0.0001:
            print(f'WARNING: alpha > 1000 or alpha < 0.0001 - alpha={a.item()}')

        self.loss_T.backward(retain_graph=True)
        if not self.args.no_ocr_loss:
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()
        if not self.args.no_ocr_loss:
            if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G):
                print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
                sys.exit()

    def backward_G_WL(self):
        self.gb_alpha = 0.7
        if self.args.is_cycle:
            self.Lcycle = self.compute_cycle_loss()

        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()

        if not self.args.no_writer_loss:
            self.loss_w_fake = self.netW(self.fake, self.input['wcl'].to(self.args.device)).mean()

        self.loss_G = self.loss_G + self.Lcycle + self.lda1 + self.lda2 - self.KLD

        if not self.args.no_writer_loss:
            self.loss_T = self.loss_G + self.loss_w_fake * self.args.writer_loss_weight
        else:
            self.loss_T = self.loss_G

        self.loss_T.backward(retain_graph=True)

        if not self.args.no_writer_loss:
            grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_WL))
            self.loss_w_fake = a.detach() * self.loss_w_fake
            self.loss_T = self.loss_G + self.loss_w_fake
        else:
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]
            a = 1
            self.loss_T = self.loss_G

        if a is None:
            print(self.loss_w_fake, self.loss_G, torch.std(grad_fake_adv))
        if a > 1000 or a < 0.0001:
            print(f'WARNING: alpha > 1000 or alpha < 0.0001 - alpha={a.item()}')

        self.loss_T.backward(retain_graph=True)

        if not self.args.no_writer_loss:
            grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_WL = 10 ** 6 * torch.mean(grad_fake_WL ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()

    def backward_G(self):
        self.opt.gb_alpha = 0.7
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake, 'z': self.z}), self.len_text_fake.detach(),
                                     self.opt.mask_loss)
        # OCR loss on real data

        pred_fake_OCR = self.netOCR(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.opt.self.args.batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size,
                                           self.len_text_fake.detach())
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

        self.loss_w_fake = self.netW(self.fake, self.wcl)
        # self.loss_OCR_fake = self.loss_OCR_fake + self.loss_w_fake
        # total loss

        # l1 = self.params[0]*self.loss_G
        # l2 = self.params[0]*self.loss_OCR_fake
        # l3 = self.params[0]*self.loss_w_fake
        self.loss_G_ = 10 * self.loss_G + self.loss_w_fake
        self.loss_T = self.loss_G_ + self.loss_OCR_fake

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]

        self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        if not False:

            self.loss_T.backward(retain_graph=True)

            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=True, retain_graph=True)[0]
            # grad_fake_wl = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]

            a = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR))

            # a0 = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_wl))

            if a is None:
                print(self.loss_OCR_fake, self.loss_G_, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
            if a > 1000 or a < 0.0001:
                print(f'WARNING: alpha > 1000 or alpha < 0.0001 - alpha={a.item()}')
            b = self.opt.gb_alpha * (torch.mean(grad_fake_adv) -
                                     torch.div(torch.std(grad_fake_adv), self.epsilon + torch.std(grad_fake_OCR)) *
                                     torch.mean(grad_fake_OCR))
            # self.loss_OCR_fake = a.detach() * self.loss_OCR_fake + b.detach() * torch.sum(self.fake)
            self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
            # self.loss_w_fake = a0.detach() * self.loss_w_fake

            self.loss_T = (1 - 1 * self.opt.onlyOCR) * self.loss_G_ + self.loss_OCR_fake  # + self.loss_w_fake
            self.loss_T.backward(retain_graph=True)
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
            self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            with torch.no_grad():
                self.loss_T.backward()
        else:
            self.loss_T.backward()

        if self.opt.clip_grad > 0:
            clip_grad_norm_(self.netG.parameters(), self.opt.clip_grad)
        if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G_):
            print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
            sys.exit()

    def optimize_D_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.optimizer_D.zero_grad()
        # if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
        self.optimizer_OCR.zero_grad()
        self.backward_D_OCR()

    def optimize_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], True)
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_OCR()

    def optimize_D(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.backward_D()

    def optimize_D_OCR_step(self):
        self.optimizer_D.step()

        self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()

    def optimize_D_OCR_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.set_requires_grad([self.netW], True)
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_D_OCR_WL()

    def optimize_D_OCR_WL_step(self):
        self.optimizer_D.step()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.step()
        self.optimizer_wl.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_wl.zero_grad()

    def optimize_D_step(self):
        self.optimizer_D.step()
        if any(torch.isnan(self.netD.infer_img.blocks[0][0].conv1.bias)):
            print('D is nan')
            sys.exit()
        self.optimizer_D.zero_grad()

    def optimize_G(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G()

    def optimize_G_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_WL()

    def optimize_G_only(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_only()

    def optimize_G_step(self):

        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

    def optimize_ocr(self):
        self.set_requires_grad([self.netOCR], True)
        # OCR loss on real data
        pred_real_OCR = self.netOCR(self.real)
        preds_size = torch.IntTensor([pred_real_OCR.size(0)] * self.opt.self.args.batch_size).detach()
        self.loss_OCR_real = self.OCR_criterion(pred_real_OCR, self.text_encode.detach(), preds_size,
                                                self.len_text.detach())
        self.loss_OCR_real.backward()
        self.optimizer_OCR.step()

    def optimize_z(self):
        self.set_requires_grad([self.z], True)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def test(self):
        self.visual_names = ['fake']
        self.netG.eval()
        with torch.no_grad():
            self.forward()

    def train_GD(self):
        self.netG.train()
        self.netD.train()
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(self.real, self.opt.self.args.batch_size)
        y = torch.split(self.label, self.opt.self.args.batch_size)
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if self.opt.toggle_grads:
            toggle_grad(self.netD, True)
            toggle_grad(self.netG, False)

        for step_index in range(self.opt.num_critic_train):
            self.optimizer_D.zero_grad()
            with torch.set_grad_enabled(False):
                self.forward()
            D_input = torch.cat([self.fake, x[counter]], 0) if x is not None else self.fake
            D_class = torch.cat([self.label_fake, y[counter]], 0) if y[counter] is not None else y[counter]
            # Get Discriminator output
            D_out = self.netD(D_input, D_class)
            if x is not None:
                pred_fake, pred_real = torch.split(D_out, [self.fake.shape[0], x[counter].shape[0]])  # D_fake, D_real
            else:
                pred_fake = D_out
            # Combined loss
            print('train_GD')
            self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(),
                                                              self.len_text.detach(), self.opt.mask_loss)
            self.loss_D = self.loss_Dreal + self.loss_Dfake
            self.loss_D.backward()
            counter += 1
            self.optimizer_D.step()

        # Optionally toggle D and G's "require_grad"
        if self.opt.toggle_grads:
            toggle_grad(self.netD, False)
            toggle_grad(self.netG, True)
        # Zero G's gradients by default before training G, for safety
        self.optimizer_G.zero_grad()
        self.forward()
        self.loss_G = loss_hinge_gen(self.netD(self.fake, self.label_fake), self.len_text_fake.detach(),
                                     self.opt.mask_loss)
        self.loss_G.backward()
        self.optimizer_G.step()

    def save_networks(self, epoch, save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    if len(self.gpu_ids) > 1:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
