import argparse
import os
import time

import numpy as np
from tqdm import tqdm
import pandas as pd

import esm
import esm.inverse_folding
from copy import deepcopy

def score_singlechain_backbones(model, alphabet, args):
    # load data
    print('Loading data and running in singlechain mode...')
    df = pd.read_csv(args.db_location, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    logps = pd.DataFrame(index=df2.index,columns=[f'esmif_monomer_full{"_masked" if args.masked else "_"}dir', f'runtime_esmif_monomer_full{"_masked" if args.masked else "_"}dir'])
    dataset = 'fireprot' if 'fireprot' in args.db_location else 's669'

    with tqdm(total=len(df2)) as pbar:
        for code, group in df2.groupby('code'):
                
                pdb_file = group['pdb_file'].head(1).item()
                code = group['code'].head(1).item()
                chain = group['chain'].head(1).item()
                print(f'Evaluating {code} {chain}')

                coords, native_seq = esm.inverse_folding.util.load_coords(pdb_file, chain)
                print('Native sequence loaded from structure file:')
                print(native_seq)

                ll_wt, _ = esm.inverse_folding.util.score_sequence(
                        model, alphabet, coords, native_seq) 
                #print('Native sequence')
                #print(f'Log likelihood: {ll:.2f}')
                #print(f'Perplexity: {np.exp(-ll):.2f}')
                for uid, row in group.iterrows():
                        try:
                                seq = row['pdb_ungapped']
                                pos = row['position']
                                ou = row['offset_up']
                                oc = int(ou) * (1 if dataset == 'fireprot' else 0)  -1
                                print(f'Mutant sequence loaded from database:')
                                print(seq)
                                print('\n')
                                start = time.time()
                                masked_coords = deepcopy(coords)
                                if args.masked:
                                    masked_coords[pos+oc] = np.inf
                                ll_mut, _ = esm.inverse_folding.util.score_sequence(
                                        model, alphabet, masked_coords, str(seq))
                                logps.at[uid, f'esmif_monomer_full{"_masked" if args.masked else "_"}_dir'] = ll_mut - ll_wt
                                logps.at[uid, f'pll_esmif_monomer_full{"_masked" if args.masked else "_"}_dir'] = np.exp(-ll_wt)
                        except Exception as e:
                                print(e)
                                print(pdb_file, chain)
                                logps.at[uid, f'esmif_monomer_full{"_masked" if args.masked else "_"}_dir'] = np.nan
                        logps.at[uid, f'runtime_esmif_monomer_full{"_masked" if args.masked else "_"}_dir'] = time.time() - start
                        pbar.update(1)
        
        # uid must be in the index col 0
        df = pd.read_csv(args.output, index_col=0)
        df = logps.combine_first(df)
        df.to_csv(args.output)


def score_multichain_backbones(model, alphabet, args):
    # load data
    print('Loading data and running in multichain mode...')
    df = pd.read_csv(args.db_location, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    logps = pd.DataFrame(index=df2.index,columns=[f'esmif_multimer_full{"_masked" if args.masked else "_"}dir', f'runtime_esmif_multimer_full{"_masked" if args.masked else "_"}dir'])

    with tqdm(total=len(df2)) as pbar:
        for code, group in df2.groupby('code'):
                
                pdb_file = group['pdb_file'].head(1).item()
                code = group['code'].head(1).item()
                chain = group['chain'].head(1).item()
                dataset = 'fireprot' if 'fireprot' in args.db_location else 's669' 

                print(f'Evaluating {code} {chain}')

                structure = esm.inverse_folding.util.load_structure(pdb_file)
                coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
                target_chain_id = chain
                native_seq = native_seqs[target_chain_id]
                print('Native sequence loaded from structure file:')
                print(native_seq)
                print('\n')

                ll_wt, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                        model, alphabet, coords, target_chain_id, native_seq) 
                #print('Native sequence')
                #print(f'Log likelihood: {ll:.2f}')
                #print(f'Perplexity: {np.exp(-ll):.2f}')

                for uid, row in group.iterrows():
                        #try:
                                seq = row['pdb_ungapped']
                                pos = row['position']
                                ou = row['offset_up']
                                oc = int(ou) * (1 if dataset == 'fireprot' else 0)  -1
                                print(f'Mutant sequence loaded from database:')
                                print(seq)
                                print('\n')
                                masked_coords = deepcopy(coords)
                                print(masked_coords[target_chain_id])
                                if args.masked:
                                    masked_coords[target_chain_id][pos+oc] = np.inf
                                start = time.time()
                                ll_mut, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                                        model, alphabet, masked_coords, target_chain_id, str(seq))
                                logps.at[uid, f'esmif_multimer_full{"_masked" if args.masked else "_"}_dir'] = ll_mut - ll_wt
                                logps.at[uid, f'pll_esmif_multimer_full{"_masked" if args.masked else "_"}_dir'] = np.exp(-ll_wt)
                        #except Exception as e:
                        #        print(e)
                        #        print(pdb_file, chain)
                        #        logps.at[uid, f'esmif_multimer_full{"_masked" if args.masked else "_"}_dir'] = np.nan
                                logps.at[uid, f'runtime_esmif_multimer_full{"_masked" if args.masked else "_"}_dir'] = time.time() - start
                                pbar.update(1)
        
        df = pd.read_csv(args.output, index_col=0)
        df = logps.combine_first(df)
        df.to_csv(args.output)

def main():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--db_location', type=str,
            help='location of the mapped database (fireprot or s669)',
    )
    parser.add_argument(
            '--output', '-o', type=str,
            help='location of the database used to store predictions.\
                  Should be a copy of the mapped database with additional cols'
    )
    parser.add_argument(
            '--structures', '-s', type=str,
            help='location of the directory containing the preprocessed structures'
    )
    parser.set_defaults(multichain_backbone=False)
    parser.add_argument(
            '--multichain_backbone', action='store_true', default=False,
            help='use the backbones of all chains in the input for conditioning'
    )
    parser.add_argument(
            '--singlechain_backbone', dest='multichain_backbone',
            action='store_false',
            help='use the backbone of only target chain in the input for conditioning'
    )
    parser.add_argument(
            '--masked', action='store_true', default=False,
            help='whether to mask the coordinates at the mutated position'
    )
    args = parser.parse_args()

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    if args.multichain_backbone:
        score_multichain_backbones(model, alphabet, args)
    else:
        score_singlechain_backbones(model, alphabet, args)

if __name__ == '__main__':
    main()
