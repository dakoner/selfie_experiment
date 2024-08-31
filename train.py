import pickle
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

device = "cuda"
dataset = "datasets/dataJ_250k_rndm_zinc_drugs_clean.csv"



# get all the inputs
num = -1
# smiles_list, smiles_alphabet, largest_smiles_len = get_smiles_encodings_for_dataset(
#     dataset
# )
# selfies_list, selfies_alphabet, largest_selfies_len = get_selfies_encodings_for_dataset(
#     smiles_list[:num], largest_smiles_len
# )


# selfies_ints = selfies_to_integers(selfies_list, selfies_alphabet, largest_selfies_len)
# data = {'selfies_alphabet': selfies_alphabet, 
#         'selfies_ints': selfies_ints,
#         'largest_selfies_len': largest_selfies_len}
# pickle.dump(data, open("selfies.pkl", "wb"))

#selfie_to_integers(selfies_list[0], symbol_to_int, largest_selfies_len)

data = pickle.load(open("selfies.pkl", "rb"))
selfies_alphabet = data['selfies_alphabet']
selfies_ints = data['selfies_ints']
largest_selfies_len = data['largest_selfies_len']

symbol_to_int = dict((c, i) for i, c in enumerate(selfies_alphabet))

pad_idx = selfies_alphabet.index("[nop]")
dataset = SelfiesDataset(device, selfies_ints)

# train_dataset = dataset[:len(dataset)//2]
# test_dataset = dataset[len(dataset)//2:]

train_dataset = dataset[:]

train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, pad_idx=selfies_alphabet.index("[nop]")),
)

# test_data_loader = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=32,
#     shuffle=True,
#     collate_fn=partial(collate_fn, pad_idx=selfies_alphabet.index("[nop]")),
# )

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


print("1")

torch.set_float32_matmul_precision('medium')
trainer = pl.Trainer(
    devices=1, accelerator="cuda", max_epochs=250
)  # , logger=wandb_logger)
trainer.fit(vae, train_data_loader)#, test_data_loader)


save_models(vae.encoder, vae.decoder, 250)


