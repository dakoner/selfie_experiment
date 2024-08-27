import os
import pandas as pd
import numpy as np
import selfies as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch


def load_models(dir_, device, epoch):
    print("loading models")
    out_dir = "./{dir}/{}".format(dir_, epoch)
    encoder = torch.load("{}/E".format(out_dir), map_location=torch.device(device))
    encoder.eval()
    decoder = torch.load("{}/D".format(out_dir), map_location=torch.device(device))
    decoder.eval()
    return encoder, decoder

def decode_to_selfie(selfies_alphabet, decoded):
    result = []
    print(decoded.shape)
    for x in decoded.tolist()[0]:
        try:
            result.append(selfies_alphabet[x])
        except IndexError:
            print("skip", x)
            pass
    mol = "".join(result)
    return mol


class SelfiesDataset(Dataset):
    def __init__(self, device, data_list):
        self.data = [torch.tensor(item).to(device) for item in data_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, *, pad_idx):
    return pad_sequence(batch, batch_first=True, padding_value=pad_idx)


def save_models(encoder, decoder, epoch):
    print("saving models")
    out_dir = "./saved_models/{}".format(epoch)
    os.makedirs(out_dir)
    torch.save(encoder, "{}/E".format(out_dir))
    torch.save(decoder, "{}/D".format(out_dir))
    
def get_smiles_encodings_for_dataset(file_path):   
    df = pd.read_csv(file_path)

    smiles_list = np.asanyarray(df.smiles)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding
    largest_smiles_len = len(max(smiles_list, key=len))

    return smiles_list, smiles_alphabet, largest_smiles_len


def get_selfies_encodings_for_dataset(smiles_list, largest_smiles_len):

    print("Largest smiles len", largest_smiles_len)
    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))
    f = open("datasets/0SelectedSMILES_QM9.sf.txt", "w")
    f.write("\n".join(selfies_list))
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    # sort symbols to get consistent alphabet between runs
    selfies_alphabet = sorted(list(all_selfies_symbols))
    selfies_alphabet.append('[nop]')

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
    print("Largest selfies len", largest_selfies_len)

    print('Finished translating SMILES to SELFIES.')
    return selfies_list, selfies_alphabet, largest_selfies_len


def selfie_to_integers(selfie, symbol_to_int, max_len):
    """Convert to a sequence of integers. Add a "start token" and "end token".
    """
    symbol_list = list(sf.split_selfies(selfie))
    return [len(symbol_to_int)] + \
        [symbol_to_int[symbol] for symbol in symbol_list] + \
        [len(symbol_to_int) + 1]


def selfies_to_integers(selfies, alphabet, max_len):
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))
    return [selfie_to_integers(selfie, symbol_to_int, max_len)
            for selfie in selfies]

