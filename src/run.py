
import os, sys, math, time, json, hashlib, random
import numpy as np, torch
import torch.nn as nn

# reproducibility
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- W&B offline so repo's wandb.Image logging works, no secret needed ----
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

# ---------- DataModule (Stefano's Zarr-RAM, verbatim) ----------
import zarr
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import tv_tensors
from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
import lightning
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def _load_zarr_to_ram(zpath, name):
    z = zarr.open(zpath, mode='r')
    imgs_z, msks_z = z['images'], z['masks']
    n = imgs_z.shape[0]
    print('[' + name + '] caricamento ' + str(n) + ' campioni in RAM...')
    imgs = np.empty(imgs_z.shape, dtype=np.uint8)
    msks = np.empty(msks_z.shape, dtype=np.uint8)
    CH = 200
    for i in range(0, n, CH):
        j = min(i + CH, n)
        imgs[i:j] = imgs_z[i:j]; msks[i:j] = msks_z[i:j]
    print('[' + name + '] fatto')
    return torch.from_numpy(imgs), torch.from_numpy(msks)

class _RAMSegDataset(TorchDataset):
    IGNORE = 255
    def __init__(self, images, masks, transforms):
        self.images = images; self.masks = masks; self.transforms = transforms
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self, idx):
        img = self.images[idx].permute(2, 0, 1).contiguous()
        mask_hw = self.masks[idx].long()
        present = torch.unique(mask_hw)
        present = present[(present != self.IGNORE) & (present < 19)]
        if present.numel() == 0:
            masks = torch.zeros((0, *mask_hw.shape), dtype=torch.bool)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            masks = (mask_hw.unsqueeze(0) == present.view(-1, 1, 1))
            labels = present.clone()
        target = {'masks': tv_tensors.Mask(masks), 'labels': labels,
                  'is_crowd': torch.zeros((labels.shape[0],), dtype=torch.bool)}
        if self.transforms is not None:
            img_t = tv_tensors.Image(img)
            img_t, target = self.transforms(img_t, target)
            img = torch.as_tensor(img_t)
        return img, target

class CityscapesSemanticRAM(LightningDataModule):
    def __init__(self, train_zarr, val_zarr, img_size=(640,640), num_classes=19,
                 batch_size=2, num_workers=8, scale_range=(0.5,2.0)):
        super().__init__(path='', batch_size=batch_size, num_workers=num_workers,
                         num_classes=num_classes, img_size=img_size,
                         check_empty_targets=False)
        self.train_zarr = train_zarr; self.val_zarr = val_zarr
        self.transforms = Transforms(img_size=img_size, color_jitter_enabled=True,
                                     scale_range=scale_range)
        self._is_setup = False
    def setup(self, stage=None):
        if self._is_setup: return self
        tr_i, tr_m = _load_zarr_to_ram(self.train_zarr, 'train')
        va_i, va_m = _load_zarr_to_ram(self.val_zarr, 'val')
        self.train_ds = _RAMSegDataset(tr_i, tr_m, self.transforms)
        self.val_ds = _RAMSegDataset(va_i, va_m, None)
        self._is_setup = True
        return self
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, drop_last=True,
                          collate_fn=self.train_collate, prefetch_factor=4,
                          **self.dataloader_kwargs)
    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, collate_fn=self.eval_collate,
                          **self.dataloader_kwargs)

# ---------- model (Stefano's) ----------
from models.vit import ViT
from models.eomt import EoMT
from training.mask_classification_semantic import MaskClassificationSemantic

def build_model():
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE)
    net = EoMT(encoder=encoder, num_classes=NUM_CLASSES, num_q=NUM_Q,
               num_blocks=NUM_BLOCKS, masked_attn_enabled=True)
    model = MaskClassificationSemantic(
        network=net, img_size=IMG_SIZE, num_classes=NUM_CLASSES,
        attn_mask_annealing_enabled=True,
        attn_mask_annealing_start_steps=[3317, 8292, 13268],
        attn_mask_annealing_end_steps=[6634, 11609, 16585],
        lr=LR, llrd=LLRD, weight_decay=WEIGHT_DECAY,
        poly_power=0.9, warmup_steps=[1000, 2000],
        ckpt_path=None, load_ckpt_class_head=False)
    return model

def load_coco_fuzzy(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = raw['state_dict'] if 'state_dict' in raw else raw
    def strip(k):
        for p in ('network.', 'model.', 'module.'):
            while k.startswith(p): k = k[len(p):]
        return k
    clean = {strip(k): v for k, v in state.items()}
    own = model.network.state_dict()
    loadable = {k: v for k, v in clean.items() if k in own and own[k].shape == v.shape}
    missing, unexpected = model.network.load_state_dict(loadable, strict=False)
    print('[fuzzy] caricate ' + str(len(loadable)) + '/' + str(len(clean)) + ' chiavi')
    print('[fuzzy] missing=' + str(len(missing)) + ' unexpected=' + str(len(unexpected)))
    return model

# ---------- LoRA (new) ----------
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

def inject_lora(net, targets, r, alpha):
    n = 0
    for _, module in net.named_modules():
        for cn, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in cn for t in targets):
                setattr(module, cn, LoRALinear(child, r, alpha)); n += 1
    return n

def apply_mode(model):
    if MODE == 'lora':
        enc = model.network.encoder
        bb = enc.backbone if hasattr(enc, 'backbone') else enc
        n = inject_lora(bb, LORA_TARGETS, LORA_R, LORA_ALPHA)
        for name, p in model.named_parameters():
            p.requires_grad = ('encoder' not in name) or ('lora_' in name)
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print('[lora] injected ' + str(n) + ' layers | trainable ' + str(tr) + '/' + str(tot))
    else:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[full] trainable ' + str(tr))
    return model

# ---------- external mIoU (Stefano's) ----------
import subprocess
def export_bin(model, out_path):
    sd = model.network.state_dict()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(sd, out_path)
    h = hashlib.sha1(open(out_path, 'rb').read()).hexdigest()
    print('[export] ' + out_path + ' SHA1=' + h)
    return h

def verify_external_miou(bin_path, tag):
    out_json = RESULTS_DIR + '/miou_ft_' + tag + '.json'
    cmd = ['python3', '-m', 'src.eval.miou_cityscapes',
           '--images-dir', CS_VAL_IMAGES, '--gt-dir', CS_VAL_GT,
           '--ckpt', bin_path, '--config', CFG_STD, '--mode', 'finetuned',
           '--num-classes', '19', '--resize', '640x640',
           '--output-json', out_json, '--seed', str(SEED)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print('STDERR:', r.stderr[-1500:]); raise RuntimeError('miou failed')
    res = json.load(open(out_json))
    print('[verify] mIoU (' + tag + ') = ' + str(round(res['miou_pct'], 2)) + '%')
    return res['miou_pct']

# ---------- trainer (CSV + wandb offline) ----------
def make_trainer(max_epochs, run_name):
    csv_logger = CSVLogger(save_dir=OUT_DIR, name='logs_' + run_name)
    wandb_logger = WandbLogger(project='eomt-local', name=run_name, save_dir=OUT_DIR, offline=True)
    run_dir = OUT_DIR + '/ckpt_' + run_name
    os.makedirs(run_dir, exist_ok=True)
    ckpt_best = ModelCheckpoint(dirpath=run_dir, filename='best',
                                monitor='metrics/val_iou_all', mode='max',
                                save_top_k=1, save_last=True)
    trainer = lightning.Trainer(max_epochs=max_epochs, precision='16-mixed',
                                accelerator='gpu', devices=1, log_every_n_steps=20,
                                callbacks=[ckpt_best], logger=[wandb_logger, csv_logger],
                                num_sanity_val_steps=0)
    return trainer, ckpt_best

# ---------- data ----------
dm = CityscapesSemanticRAM(train_zarr=TRAIN_ZARR, val_zarr=VAL_ZARR,
                           img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                           batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
dm.setup()
print('DataModule pronto. Train:', len(dm.train_ds), 'Val:', len(dm.val_ds))

# ---------- sanity (5 epochs) ----------
print('\n===== SANITY (' + MODE + ') =====')
model = build_model(); model = load_coco_fuzzy(model, CKPT_COCO_SRC); model = apply_mode(model)
st, _ = make_trainer(SANITY_EPOCHS, MODE + '_sanity')
st.fit(model, datamodule=dm)
bs = OUT_DIR + '/eomt_ft_' + MODE + '_sanity.bin'
export_bin(model, bs)
ms = verify_external_miou(bs, MODE + '_sanity')
assert ms >= SANITY_MIOU_GATE, 'gate failed: ' + str(ms)

# ---------- full (50 epochs) ----------
print('\n===== FULL (' + MODE + ') =====')
mf = build_model(); mf = load_coco_fuzzy(mf, CKPT_COCO_SRC); mf = apply_mode(mf)
ft, _ = make_trainer(FULL_EPOCHS, MODE + '_full')
ft.fit(mf, datamodule=dm)
bf = OUT_DIR + '/eomt_ft_' + MODE + '.bin'
shf = export_bin(mf, bf)
mio = verify_external_miou(bf, MODE + '_full')
print('\n' + '='*55)
print('RESULT (' + MODE + '): mIoU=' + str(round(mio,2)) + '% SHA1=' + shf)
print('checkpoint:', bf)
print('loss csv:', OUT_DIR + '/logs_' + MODE + '_full/version_0/metrics.csv')
print('='*55)
