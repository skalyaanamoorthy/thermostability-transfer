import torch

import argparse
import time

import numpy as np
from tqdm import tqdm
import pandas as pd

import esm
import esm.inverse_folding

from copy import deepcopy


def assert_diff_by_one(str1, str2):
    assert len(str1) == len(str2), "Strings are not of same length"
    diff_count = 0
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            diff_count += 1
    assert diff_count == 1, "Strings are not different by exactly one character"

def score_backbones(model, alphabet, args):
    # load data
    suffix = f'{"multimer" if args.multimer else "monomer"}{"_masked" if args.mask_coords else ""}_dir'
    print(f'Loading data and running in {suffix} mode...')
    df = pd.read_csv(args.db_loc, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    dataset = args.dataset

    device = 'cuda:0'
    model = model.to(device)

    logps = pd.DataFrame(index=df2.index,columns=[f'esmif_{suffix}', f'runtime_esmif_{suffix}'])

    with tqdm(total=len(df2)) as pbar:
        for code, group in df2.groupby('code'):
                
            pdb_file = group['pdb_file'].head(1).item()
            code = group['code'].head(1).item()
            chain = group['chain'].head(1).item()
            print(f'Evaluating {code} {chain}')

            for uid, row in group.iterrows():
                with torch.no_grad():
                    
                    start = time.time()
                    pos = row['position']
                    wt = row['wild_type']
                    mut = row['mutation']
                    ou = row['offset_up']
                    seq = row['pdb_ungapped']
                    print(seq)
                    oc = int(ou) * (1 if dataset == 'fireprot' else 0)  -1

                    if args.multimer:
                        structure = esm.inverse_folding.util.load_structure(pdb_file)
                        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
                        target_chain_id = chain
                        coords = esm.inverse_folding.multichain_util._concatenate_coords(coords, target_chain_id)
                        native_seq = native_seqs[target_chain_id]
                    else:
                        coords, native_seq = esm.inverse_folding.util.load_coords(pdb_file, chain)
                    masked_coords = deepcopy(coords)
                    if args.mask_coords:
                        masked_coords[pos+oc] = np.inf
                    coords = masked_coords
                    print('Native sequence loaded from structure file:')
                    print(native_seq)
                    try:
                        assert_diff_by_one(native_seq, seq)
                    except Exception as e:
                        print(e)

                    masked_seq = list(native_seq)
                    masked_seq[pos+oc] = '<mask>'
                    masked_seq = ''.join(masked_seq)

                    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
                    batch = [(coords, None, masked_seq)]
                    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)

                    prev_output_tokens = tokens[:, :-1]
                    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
                    ll = (logits[0][alphabet.all_toks.index(mut)][pos+oc] - logits[0][alphabet.all_toks.index(wt)][pos+oc]).item()

                    logps.at[uid, f'esmif_{suffix}'] = ll
                    logps.at[uid, f'runtime_esmif_{suffix}'] = time.time() - start
                    pbar.update(1)
        
        # uid must be in the index col 0
        df = pd.read_csv(args.output, index_col=0)
        df = logps.combine_first(df)
        df.to_csv(args.output)


def score_backbones_inverse(model, alphabet, args):
    suffix = f'{"multimer" if args.multimer else "monomer"}{"_masked" if args.mask_coords else ""}_inv'
    print(f'Loading data and running in {suffix} mode...')
    df = pd.read_csv(args.db_loc, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    dataset = args.dataset

    device = 'cuda:0'
    model = model.to(device)

    logps = pd.DataFrame(index=df2.index,columns=[f'esmif_{suffix}', f'runtime_esmif_{suffix}'])
    
    with tqdm(total=len(df2)) as pbar:
        for uid, row in df2.iterrows():
            with torch.no_grad():
                try:
                    pdb_file = row['mutant_pdb_file']
                    code = row['code']
                    chain = 'A' #row['chain']  
                    print(f'Evaluating {code} {chain}')

                    pos = row['position']
                    wt = row['wild_type']
                    mut = row['mutation']
                    ou = row['offset_up']
                    ro = row['offset_robetta']
                    ro = 0
                    seq = row['pdb_ungapped']
                    print(seq)
                    start = time.time()
                    oc = int(ou) * (1 if dataset == 'fireprot' else 0)  -1 -int(ro)

                    if args.multimer:
                        structure = esm.inverse_folding.util.load_structure(pdb_file)
                        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
                        target_chain_id = chain
                        mutant_seq = native_seqs[target_chain_id]
                    else:
                        coords, mutant_seq = esm.inverse_folding.util.load_coords(pdb_file, chain)
                    masked_coords = deepcopy(coords)
                    if args.mask_coords:
                        masked_coords[pos+oc] = np.inf
                    coords = masked_coords                       
                    print('Mutant sequence loaded from structure file:')
                    print(mutant_seq)
                    #assert mutant_seq == seq, 'Provided sequence does not match structure'

                    masked_seq = list(mutant_seq)
                    masked_seq[pos+oc] = '<mask>'
                    masked_seq = ''.join(masked_seq)

                    start = time.time()
                    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
                    batch = [(coords, None, masked_seq)]
                    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
                    prev_output_tokens = tokens[:, :-1]
                    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
                    ll = (logits[0][alphabet.all_toks.index(wt)][pos+oc] - logits[0][alphabet.all_toks.index(mut)][pos+oc]).item()

                    logps.at[uid, f'esmif_{suffix}'] = ll

                except Exception as e:
                        print(e)
                        print(pdb_file, chain)
                        logps.at[uid, f'esmif_{suffix}'] = np.nan

                logps.at[uid, f'runtime_esmif_{suffix}'] = time.time() - start
                pbar.update(1)

        df = pd.read_csv(args.output, index_col=0)
        df = logps.combine_first(df)
        df.to_csv(args.output)

def main():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--db_loc', type=str,
            help='location of the mapped database (file name should contain fireprot or s669)',
    )
    parser.add_argument(
            '--output', '-o', type=str,
            help='location of the database used to store predictions.\
                  Should be a copy of the mapped database with additional cols'
    )
    parser.add_argument(
            '--mask_coords', action='store_true', default=False,
            help='whether to mask the coordinates at the mutated position'
    )
    parser.set_defaults(multichain_backbone=False)
    parser.add_argument(
            '--multimer', action='store_true', default=False,
            help='use the backbones of all chains in the input for conditioning'
    )
    parser.add_argument(
            '--inverse', action='store_true', default=False,
            help='use the mutant structure and apply a reversion mutation'
    )
    args = parser.parse_args()

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    if 'fireprot' in args.db_loc.lower():
        args.dataset = 'fireprot'
    elif 's669' in args.db_loc.lower() or 's461' in args.db_loc.lower():
        args.dataset = 's669'
    else:
        print('Inferred use of user-created database')
        args.dataset = 'custom'

    if args.inverse:
        score_backbones_inverse(model, alphabet, args)
    else:
        score_backbones(model, alphabet, args)

if __name__ == '__main__':
    main()
