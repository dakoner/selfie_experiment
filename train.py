import torch
from functools import partial
import pytorch_lightning as pl
from utils import (
    get_smiles_encodings_for_dataset,
    selfies_to_integers,
    get_selfies_encodings_for_dataset,
    selfie_to_integers,
    save_models,
    collate_fn,
    SelfiesDataset,
    decode_to_selfie
)
from VAE import VAE

device = "cpu"




# get all the inputs
num = -1
smiles_list, smiles_alphabet, largest_smiles_len = get_smiles_encodings_for_dataset(
    "datasets/0SelectedSMILES_QM9.txt"
)
selfies_list, selfies_alphabet, largest_selfies_len = get_selfies_encodings_for_dataset(
    smiles_list[:num], largest_smiles_len
)

print(selfies_alphabet)
selfies_ints = selfies_to_integers(selfies_list, selfies_alphabet, largest_selfies_len)

symbol_to_int = dict((c, i) for i, c in enumerate(selfies_alphabet))
selfie_to_integers(selfies_list[0], symbol_to_int, largest_selfies_len)

pad_idx = selfies_alphabet.index("[nop]")
dataset = SelfiesDataset(
    selfies_ints,
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, pad_idx=selfies_alphabet.index("[nop]")),
)

vae = VAE(
    len(selfies_alphabet) + 2,
    selfies_alphabet.index("[nop]"),
    q_dim=512,
    q_layers=3,
    latent_dim=512,
    emb_dim=512,
    d_dim=512,
    d_layers=3,
    lr=2e-4,
).to(device)


data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=partial(collate_fn, pad_idx=selfies_alphabet.index("[nop]")),
)
print("1")

torch.set_float32_matmul_precision('medium')
trainer = pl.Trainer(
    devices=1, accelerator="cuda", max_epochs=100
)  # , logger=wandb_logger)
trainer.fit(vae, data_loader)


save_models(vae.encoder, vae.decoder, 100)


