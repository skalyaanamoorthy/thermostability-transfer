import torch
import os
import argparse
import time
import string
import itertools
import glob
import re

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from typing import List, Tuple
import numpy as np
import collections.abc

collections.Iterable = collections.abc.Iterable

from evcouplings import align


def subsample(infile, nseqs, reps):
        aln = align.Alignment.from_file(open(infile, 'r'), format='a3m')
        aln.set_weights()
        with open('neff.csv', 'a') as f:
            f.write(f"{infile.split('/')[-1].split('_')[0]}, {sum(aln.weights)}\n")
        outfolder = '/'.join(infile.split('/')[:-1]) + '/subsampled/'
        os.makedirs(outfolder, exist_ok=True)

        if len(aln) < nseqs:
            print(len(aln))
            print('Returning all sequences as there were not enough for subsampling!')
            seqs = []
            match_cols = np.where(aln[0]!='-')
            for s in range(len(aln)):
                seqs.append((s, ''.join(aln[s][match_cols])))

            seqs[0] = (0, ''.join(aln[0][match_cols]))

            outfile = os.path.join(outfolder, infile.split('/')[-1].replace('.a3m', f'_reduced_subsampled_0.a3m'))
            align.write_a3m(seqs, open(outfile, 'w'))
            return

        for i in range(reps):
                selected = np.random.choice(range(len(aln)), size=nseqs, p=aln.weights/sum(aln.weights), replace=False)
                selected = np.sort(selected)
                seqs = []
                match_cols = np.where(aln[0]!='-')
                for s in selected:
                    seqs.append((s, ''.join(aln[s][match_cols])))

                seqs[0] = (0, ''.join(aln[0][match_cols]))

                outfile = os.path.join(outfolder, infile.split('/')[-1].replace('.a3m', f'_reduced_subsampled_{i}.a3m'))
                align.write_a3m(seqs, open(outfile, 'w'))


#def remove_insertions(sequence: str) -> str:
#    """ 
#    Removes any insertions into the sequence. Needed to load aligned sequences 
#    in an MSA. 
#    """
#    # This is an efficient way to delete lowercase characters and insertion characters from a string
#    deletekeys = dict.fromkeys(string.ascii_lowercase)
#    deletekeys["."] = None
#    deletekeys["*"] = None
#    translation = str.maketrans(deletekeys)
#    return sequence.translate(translation)


def read_msa(filename: str, nseq: int, start: int, end: int):
    """ 
    Reads the first nseq sequences from an MSA file, automatically removes 
    insertions, and slices from the start to the end column. The input file must 
    be in a3m format (although we use the SeqIO fasta parser) for 
    remove_insertions to work properly.
    """
    msa = [(record.description, str(record.seq[start:end])) #record.seq[start:end]
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
          ]
    return msa


def score_sequences(args):

    df = pd.read_csv(args.db_location, index_col=0).reset_index()
    df2 = df.groupby('uid').first()
    dataset = 'fireprot' if 'fireprot' in args.db_location else 's669'

    logps = pd.DataFrame(index=df2.index,columns=[f'msa_{i+1}_dir' for i in range(5)] + [f'runtime_msa_{i+1}_dir' for i in range(5)])

    model, alphabet = pretrained.load_model_and_alphabet(f'esm_msa1b_t12_100M_UR50S')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    with tqdm(total=len(df2['code'].unique())) as pbar:
        for code, group in df2.groupby('code'):
            msas = os.path.join(args.alignments, 'subsampled/')
            os.makedirs(msas, exist_ok=True)

            sequence = group.head(1)['uniprot_seq'].item()
            orig_msa = group.head(1)['msa_file'].item()
            subsample(orig_msa, nseqs=384, reps=5)

            for i in range(5):
                
                msa = os.path.join(msas, orig_msa.split('/')[-1].replace('.a3m', f'_reduced_subsampled_{i}.a3m'))
                print(msa)
                if not os.path.exists(msa):
                    print(f'Only 1 subsampled alignment for {code}, skipping to next protein')
                    continue
        
                for uid, row in group.iterrows():
                    with torch.no_grad():
                        try:
                            pos = row['position']
                            wt = row['wild_type']
                            mt = row['mutation']
                            ou = row['offset_up']
                            ws = row['window_start']
                            sequence = row['uniprot_seq'][ws:ws+1022]
                            #msa = row['msa_filename']
                            oc = int(ou) * (0 if dataset == 'fireprot' else -1)  -1 -ws
                            idx = pos + oc

                            start = time.time()

                            batch_converter = alphabet.get_batch_converter()
                            # msas in repo have already been reduced
                            data = [read_msa(msa, 384, ws, ws+1022)]
                            batch_labels, batch_strs, batch_tokens = batch_converter(data)

                            batch_tokens_masked = batch_tokens.clone()
                            batch_tokens_masked[0, 0, idx + 1] = alphabet.mask_idx
                            with torch.no_grad():
                                token_probs = torch.log_softmax(
                                    model(batch_tokens_masked.cuda())["logits"], dim=-1
                                )
                            token_probs = token_probs[:, 0, :]
                            #print(token_probs.shape)
                            assert sequence[idx] == wt

                            wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
                            score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]

                            logps.at[uid, f'msa_{i+1}'] = score.item()
                            logps.at[uid, f'runtime_msa_{i+1}'] = time.time() - start
                        except Exception as e:
                            print(e, code, wt, pos, mt)
                            logps.at[uid, f'msa_{i+1}_dir'] = np.nan
                            logps.at[uid, f'runtime_msa_{i+1}_dir'] = np.nan
            pbar.update(1)
                    
    logps['msa_transformer_median_dir'] = logps[[f'msa_{i+1}_dir' for i in range(5)]].median(axis=1)
    logps['msa_transformer_mean_dir'] = logps[[f'msa_{i+1}_dir' for i in range(5)]].mean(axis=1)
    logps['runtime_msa_transformer_median_dir'] = logps[[f'runtime_msa_{i+1}_dir' for i in range(5)]].sum(axis=1)
    logps['runtime_msa_transformer_mean_dir'] = logps['runtime_msa_transformer_median_dir']
    logps.to_csv('msa_transformer_preds.csv')
    df = pd.read_csv(args.output, index_col=0)
    df = logps.combine_first(df)
    df.to_csv(args.output)


def main():
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--db_location', type=str,
            help='location of the mapped database (file name should contain fireprot or s669)',
    )
    parser.add_argument(
            '--alignments', '-a', type=str,
            help='directory where alignments are stored. There should be only\
                one file matching the pattern CODE_*.a3m where CODE is the pdb\
                id of the structure the alignment is based upon'
    )
    parser.add_argument(
            '--output', '-o', type=str,
            help='location of the database used to store predictions.\
                  Should be a copy of the mapped database with additional cols'
    )
    #parser.add_argument(
    #        '--do_subsampling', action='store_true', help='whether to subsample the MSA down to 384',
    #)
    args = parser.parse_args()

    score_sequences(args)

if __name__ == '__main__':
    main()
