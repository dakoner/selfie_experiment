import numpy as np
import torch
from VAE import VAE, VAEEncoder, VAEDecoder
from utils import (
    get_smiles_encodings_for_dataset,
    selfies_to_integers,
    get_selfies_encodings_for_dataset,
    selfie_to_integers,
    decode_to_selfie,
    load_models
)
import selfies as sf

device = "cuda"

num = -1
smiles_list, smiles_alphabet, largest_smiles_len = get_smiles_encodings_for_dataset(
    "datasets/0SelectedSMILES_QM9.txt"
)
selfies_list, selfies_alphabet, largest_selfies_len = get_selfies_encodings_for_dataset(
    smiles_list[:num], largest_smiles_len
)

print(selfies_alphabet)
selfies_ints = selfies_to_integers(selfies_list, selfies_alphabet, largest_selfies_len)

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

vae.encoder, vae.decoder = load_models(100)

# # Example decoding
# idx = 0

# input_tensor = torch.nn.functional.pad(
#     torch.Tensor(selfies_ints[idx]).unsqueeze(0).long(),
#     pad=(0, largest_selfies_len - len(selfies_ints[idx])),
#     value=selfies_alphabet.index("[nop]"),
# ).to(device)
# print(input_tensor.shape)
# latent, *_ = vae.encoder(input_tensor)

# # Since we don't a priori know the length, just use something longer than the training data
# decoded, _ = vae.decoder(latent.repeat(1, largest_selfies_len, 1))
# # back to indices
# print(decoded.shape)
# print(selfies_list[idx])
# print(input_tensor)
# print(decode_to_selfie(selfies_alphabet, input_tensor))
# print(decoded.argmax(-1))
# print(decode_to_selfie(selfies_alphabet, decoded.argmax(-1)))
# print(vae.accuracy(input_tensor, decoded.argmax(-1)))

mol1 = "[N][N][=C][C][=N][Ring1][Branch1]"
mol2 = "[C][C][C][=Branch1][C][=O][C][C]"
idx1 = selfies_list.index(mol1)
idx2 = selfies_list.index(mol2)

# make it a tensor and fix input dimensions
print(selfies_list[idx1], sf.decoder(selfies_list[idx1]))
input_tensor = torch.nn.functional.pad(
    torch.Tensor(selfies_ints[idx1]).unsqueeze(0).long(),
    pad=(0, largest_selfies_len - len(selfies_ints[idx1])),
    value=selfies_alphabet.index("[nop]"),
).to(device)
latent1, *_ = vae.encoder(input_tensor)

# make it a tensor and fix input dimensions
input_tensor = torch.nn.functional.pad(
    torch.Tensor(selfies_ints[idx2]).unsqueeze(0).long(),
    pad=(0, largest_selfies_len - len(selfies_ints[idx2])),
    value=selfies_alphabet.index("[nop]"),
).to(device)
latent2, *_ = vae.encoder(input_tensor)

points = np.linspace(latent1.detach().cpu().numpy(), latent2.detach().cpu().numpy(), 50)
for point in points:
    decoded, _ = vae.decoder(torch.from_numpy(point).to(device).repeat(1, largest_selfies_len, 1))
    # back to indices

    decoded.argmax(-1)
    result = []
    for x in decoded.argmax(-1).tolist()[0]:
        try:
            result.append(selfies_alphabet[x])
        except IndexError:
            pass
    mol = "".join(result)
    print(mol, sf.decoder(mol))
    print()
