"""
This script is used to preprocess data from FireProtDB or S669. Since S461 is a
subset of S669, this script will preprocess all data needed for S461. The script 
pulls structure and sequence data from PDB and UniProt. At the end of 
preprocessing, the tabular data originating from the respective databases is 
ready for prediction by other tools.
"""

import os
import argparse
import utils
import glob
import re

import pandas as pd
import numpy as np

# Some methods want unusual residues mapped to their closest original residue.
# Others want them removed or replaced with an 'X'. This mapping will be used to
# find the appropriate replacement later

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 
    'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 
    'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 
    'TYR': 'Y', 'MET': 'M','MSE': 'Z', 'UNK': '9', 'X': 'X'} 

#canonical = {'MSE': 'MET'}

def main(args):

    output_path = os.path.abspath(args.output_root)
    print('Full path to outputs:', output_path)

    # used when running preprocessing on a different computer than inference
    if args.internal_path is not None:
        internal_path = args.internal_path
    else:
        internal_path = output_path
    
    BIO_ASSEMBLIES_DIR = os.path.join(output_path, 'assemblies')
    STRUCTURES_DIR = os.path.join(output_path, 'structures')
    ALIGNMENTS_DIR = os.path.join(output_path, 'alignments')
    SEQUENCES_DIR = os.path.join(output_path, 'sequences')
    WINDOWS_DIR = os.path.join(output_path, 'windows')
    # this defines where results from Rosetta will be stored, for organization
    RESULTS_DIR = os.path.join(output_path, 'results')
    DATA_DIR = os.path.join(output_path, 'data')

    # first build a folder structure for organizing inputs and outputs.
    for folder in [BIO_ASSEMBLIES_DIR, STRUCTURES_DIR, ALIGNMENTS_DIR, 
                   SEQUENCES_DIR, WINDOWS_DIR, RESULTS_DIR, DATA_DIR,
                   os.path.join(SEQUENCES_DIR, 'fasta_wt'), 
                   os.path.join(SEQUENCES_DIR, 'fasta_mut'),
                   os.path.join(SEQUENCES_DIR, 'fasta_up'),
                   os.path.join(output_path, 'DMS_Tranception'),
                   os.path.join(output_path, 'DMS_MSA')]:
        os.makedirs(folder, exist_ok=True)

    # chains listed in databases do not always correspond to the biological
    # assembly's naming convention.
    wrong_chains = {'1ACB': 'I', '1AON': 'O', '1AYF': 'B', '1ANT': 'L',
                    '1GUY': 'C', '1HK0': 'X', '1HYN': 'P', '1RN1': 'C',
                    '1RTP': '1', '1ZNJ': 'B', '2CI2': 'I', '5CRO': 'O',
                    '1IV7': 'A', '3L15': 'A', '4N6V': '2', '1YU5': 'X',
                    '2L7F': 'P', '2L7M': 'P', '2LVN': 'C'}
    
    orig_chains = {'1IV7': 'B', '3L15': 'B', '4N6V': '0'}

    # original database needs to be at this location and can be obtained from
    # the FireProtDB website or from Pancotti et al.
    db = pd.read_csv(args.db_loc)
    if 'fireprot' in args.db_loc.lower():
        dataset = 'fireprot'
        # some entries in FireProt do not have associated structures
        db = db.dropna(subset=['pdb_id'])
        # get the first PDB from the list (others might be alternate structures)
        db['code'] = db['pdb_id'].apply(lambda x: x.split('|')[0])
        # assign a unique identifier for matching tables later on
        db['uid'] = db['code']+ '_' + \
            db['position'].astype(str) + db['mutation']
    elif 's669' in args.db_loc.lower():
        dataset = 's669'
        db['code'] = db['Protein'].str[0:4]
        db['chain'] = db['Protein'].str[-1]
        # correct one incorrect record (authors used a modified structure)
        db.loc[(db['Mut_seq']=='Y200H') & (db['code']=='3L15'), 'Mut_seq'] = \
            'Y189H'
        # assign a unique identifier for matching tables later on
        db['uid'] = db['Protein'].str[:4] + '_' + db['Mut_seq'].str[1:]
    else:
        print('Running with a custom user-specified database\n' 
              'This is NOT desired behaviour for reproducing benchmarks')
        db['code'] = db['code'].str.upper()
        for c in ['code', 'chain', 'wild_type', 'position', 'mutation']:
            assert c in db.columns
        if not 'uid' in db.columns:
            db['uid'] = db['code'] + '_' \
                + db['position'].astype(str) + db['mutation']
        dataset = args.dataset

    hit = pd.DataFrame() # collection of successfully parsed mutations

    # mutations which were not successfully parsed
    miss = pd.DataFrame(columns=['code', 'wt', 'pos', 'mut']) 

    # iterate through one PDB code at a time, e.g. all sharing the wt structure
    for code, group in db.groupby('code'):

        if code in ['2MW9', '2MWA', '2MWB']:
            print('Skipping', code)
            continue

        print(f"Parsing {code}, {len(group['uid'].unique())} unique mutations")

        # chains listed in database do not always correspond to the assembly
        if code in wrong_chains:
            chain = wrong_chains[code]
        else:
            chain = str(group['chain'].head(1).item())
        
        # directory which will be used to organize RESULTS structure-wise
        os.makedirs(
            os.path.join(RESULTS_DIR, f'{code}_{chain}'), exist_ok=True
            )

        # get the biological assembly, which includes multimeric structures
        prot_path, prot_file = utils.download_assembly(
            code, chain, BIO_ASSEMBLIES_DIR
            )

        if dataset == 'fireprot':
            # in the FireProtDB, the UniProt sequence is provided
            uniprot_seq = group['sequence'].head(1).item()
            with open(
                os.path.join(SEQUENCES_DIR, 'fasta_up', f'{code}_{chain}.fa'), 
            'w') as f:
                f.write(f'>{code}_{chain}\n{uniprot_seq}')

        # assume we need to get the uniprot sequence corresponding to the entry
        else:
            uniprot_seq, accession = utils.get_uniprot_s669(
                code, chain, SEQUENCES_DIR
                )
        
        # get the pdb sequence corresponding to the entry
        chains, is_nmr = utils.extract_structure(
            code, chain, d, prot_path, prot_file, STRUCTURES_DIR
            )
        pdb_seq = chains[chain]
        multimer = len(chains.keys())
        
        # align the pdb sequence to the uniprot sequence
        alignment_df, window_start, pdb_ungapped, uniprot_seq = \
            utils.align_sequence_structure(
                code, chain, pdb_seq, dataset,
                SEQUENCES_DIR, WINDOWS_DIR, ALIGNMENTS_DIR, uniprot_seq
                )

        # create a convenience link to the alignment file
        # note: this MSA (included in the repo) is already reduced
        # to the context of interest and has no more than 90% identity
        # between sequences and no less than 75% coverage per sequence
        matching_files = glob.glob(
            os.path.join(args.alignments, f'{code}_*.a3m')
            )
        assert len(matching_files) <= 1, \
            f"Expected one file, but found {len(matching_files)}"
        if len(matching_files) == 0:
            exp = "un" if code not in ["1DXX", "1TIT", "1JL9"] else ""
            print(f'Did not find an MSA for {code}. This is {exp}expected')
            orig_msa = ''
        else:
            orig_msa = os.path.abspath(matching_files[0])

        # need the original chain to refer to predicted structures  
        chain_orig = orig_chains[code] \
            if code in orig_chains.keys() else chain

        # create a convenience link to the structure file
        pdb_file = os.path.join(STRUCTURES_DIR, f'{code}_{chain}.pdb')

        inferred_offset = 0

        if args.infer_pos:
            offsets = []
            for name, group2 in group.groupby(
                ['wild_type', 'position', 'mutation']
                if dataset != 's669' else 'Mut_seq'
                ):
                wt, pos, mut = name
                try:
                    pos_ = utils.infer_pos(group2, pdb_seq)
                    inferred_offset = pos - pos_
                    offsets.append(inferred_offset)
                except:
                    #print(name)
                    continue
            # hacky solution
            inferred_offset = int(np.median(offsets))

        # now we process and validate individual mutations from the database
        for name, group2 in group.groupby(
            ['wild_type', 'position', 'mutation']
            if dataset != 's669' else 'Mut_seq'
            ):

            # get the wild-type amino acid, 
            # the position of the mutation, 
            # and the mutant residue identity one-letter code

            if dataset == 's669':
                wt = name[0]
                pos = int(name[1:-1])
                mut = name[-1]
            else:
                wt, pos, mut = name
                pos -= inferred_offset

            # get offsets for interconversion between uniprot and pdb positions
            offset_up = utils.get_offsets(wt, pos, dataset, alignment_df)

            # fireprot mutation labels correspond to uniprot entries; 
            # offset_up maps these to the pdb sequence
            if dataset == 'fireprot':
                offset_pdb = offset_up
            # s669 mutation labels correspond to pdb entries; 
            # offset_up is used (in reverse) to map to uniprot
            else:
                # no offset since we are operating in pdb coordinates
                offset_pdb = 0

            # still validate offset and ensure it correctly maps to uniprot
            # the -1 converts to zero-based indexing
            if uniprot_seq[
                int(pos) - 1 - (offset_up if dataset!='fireprot' else 0)
                ] != wt:
                print(code, wt, pos, mut, 
                'uniprot wt does not match with provided mutation'
                ) # happens rarely when the structure is already mutated

            # attempt to validate the offsets and hence ensure mutations are 
            # being assigned to the correct location
            try:
                # '9' is unknown, but it needed to be distinct from residues
                # such as MSE which are only sometimes unknown
                pu = pdb_ungapped.replace('9', 'X')
                # ESM-IF canonicalizes by default  
                pu = pu.replace('Z', 'M')
                
                # validation step
                assert pu[int(pos) - 1 + offset_pdb] == wt, \
                    f'PDB/UniProt offset is {offset_pdb}'

                # format the mutant sequence as required by each method
                pu = utils.generate_mutant_sequences(
                    code, chain, pos, mut, pu, offset_pdb, SEQUENCES_DIR
                    )

                # predicted mutant structures obtained from Pancotti et al.
                mutant_pdb_file = os.path.join(
                    output_path, 'structures_mut',
                    f'{code.lower()}{chain_orig}_{wt}{pos}{mut}.pdb')
                # difference in numbering for this one case
                if code == '3L15':
                    mutant_pdb_file = os.path.join(
                        output_path, 'structures_mut',
                        f'{code.lower()}{chain_orig}_{wt}200{mut}.pdb')

                pdb_file = re.sub(output_path, internal_path, pdb_file)
                mutant_pdb_file = re.sub(
                    output_path, internal_path, mutant_pdb_file
                    )
                orig_msa = re.sub(output_path, internal_path, orig_msa)

                new_hit = pd.DataFrame({0: {
                    'code':code, 'wild_type':wt, 'position':pos, 'mutation':mut,
                    'chain':chain, 'offset_up':offset_up, 'is_nmr':is_nmr,
                    # compensate numbering error
                    'offset_robetta': '-11' if code=='3L15' else '0',
                    'pdb_ungapped': pu, 'uid': code + '_' + str(pos+inferred_offset) + mut, 
                    'uniprot_seq': uniprot_seq, 'window_start': window_start, 
                    'pdb_file': pdb_file,'mutant_pdb_file': mutant_pdb_file,
                    'multimer': multimer, 'msa_file': orig_msa, 
                    'tranception_dms': os.path.join(internal_path,
                    'DMS_Tranception', f'{code}_{chain}_{dataset}.csv'),
                    'inferred_offset': inferred_offset
                }}).T

                # used for getting features in features.py
                if dataset == 's669':
                    new_hit['uniprot_id'] = accession

                # ultimately turns into the output table used downstream
                hit = pd.concat([hit, new_hit])

            # there are exceptions in FireProt where the PDB sequence doesn't 
            # match UniProt at mutated positions, e.g. due to mutant structures
            except Exception as e:
                print(e, code, wt, pos, mut)
                miss = pd.concat([miss, pd.DataFrame({'0': {
                    'code':code, 'wt': wt, 'pos': pos, 'mut': mut, 
                    'ou': offset_up  
                }}).T])

    if args.verbose:
        hit.to_csv(
            os.path.join(output_path, DATA_DIR, f'hit_{dataset}.csv')
            )
        miss.to_csv(
            os.path.join(output_path, DATA_DIR, f'miss_{dataset}.csv')
            )

    # remove overlapping columns
    if dataset == 's669':
        db = db.drop(['code', 'chain'], axis=1)
    else:
        db = db.drop(
            ['code', 'chain', 'wild_type', 'position', 'mutation'], axis=1
            )

    # combine all the original mutation information from FireProtDB with hits
    out = db.merge(hit, on=['uid'])
    # check how many mutants could not be processed or validated
    print('Unique mutants lost from original dataset:', len(db)-len(out))
    #print(set(out['uid']).difference(set(db['uid'])))

    # at this point, we have all the information about the mapping between 
    # sequence and structure, and we have validated the mutant sequences.
    # Now we just need to prepare the input files for each predictor based on 
    # the format it expects. This usually includes sequence, the location of the 
    # mutation, and the structure file location for structural methods.
    
    # this next section is for compatibility with Rosetta
    if args.rosetta:
        hit_rosetta = pd.DataFrame()
    # robetta is referring to the modelled structures for inverse mutations
    if args.inverse:
        hit_robetta = pd.DataFrame()

    # iterate back through the output dataframe based on wt structure
    for code, group in out.groupby('code'):

        chain = group['chain'].head(1).item()

        # save the data in a method specific directory in the output_root 
        # e.g. DMS_MSA for MSA transformer
        utils.save_formatted_data(
            code, chain, group, dataset, output_path
            )
        
        # make sure Rosetta's parsing doesn't mess up the alignment
        # happens due to residue deletions in 1AYE, 1C52, 1CTS, 5AZU
        if args.rosetta:
            offsets_rosetta = utils.get_rosetta_mapping(
                code, chain, group, dataset, SEQUENCES_DIR, RESULTS_DIR
                )
            # would only be None if the structure or chain was not found
            if offsets_rosetta is not None:
                hit_rosetta = pd.concat([hit_rosetta, offsets_rosetta])
        # do the same for inverse structures
        if args.inverse and args.rosetta:
            offsets_robetta = utils.get_rosetta_mapping(
                code, chain, group, dataset, 
                SEQUENCES_DIR, RESULTS_DIR, inverse=True
                )
            if offsets_robetta is not None:
                hit_robetta = pd.concat([hit_robetta, offsets_robetta])

    # combine Rosetta offsets with all other data
    if args.rosetta:
        out = out.merge(
            hit_rosetta, 
            on=['code', 'chain', 'position', 'mutation'], 
            how='left'
            )
    if args.inverse and args.rosetta:
        out = out.merge(
            hit_robetta, 
            on=['code', 'chain', 'position', 'mutation'], 
            how='left'
            )
        
    out = out.set_index('uid')
    
    # this is the main input file for all PSLMs
    out.to_csv(os.path.join(output_path, DATA_DIR, 
        f'{dataset}_mapped.csv'))

    grouped = out.groupby('uid').first()[
        ['code', 'chain', 'wild_type', 'position', 'mutation', 'offset_up'] \
        + (['offset_rosetta'] if args.rosetta else [])
        ]

    # this file is used by Rosetta to determine where mutations should be made
    if args.rosetta:
        grouped['offset_rosetta'] = \
            grouped['offset_rosetta'].fillna(0).astype(int)
    grouped.to_csv(
        os.path.join(output_path, DATA_DIR,
            f'{dataset}_unique_muts_offsets.csv')
            )

    # finally, this file is used as an input for a batch job on an HPC
    # these indicies are associated with the unique entries, which can be
    # used to index individual rows of the dataframe for running Rosetta
    # relaxation in parallel
    with open(os.path.join(
        output_path, DATA_DIR, f'{dataset}_rosetta_indices.txt'
        ), 'w') as f:
        inds = ','.join(grouped.reset_index(drop=True).reset_index()\
            .groupby(['code', 'chain']).first()['index'].astype(str))
        print(inds)
        f.write(inds)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    description = 'Preprocesses data from either s669 or \
                                   FireProtDB for downstream prediction'
                    )
    parser.add_argument('--dataset', help='name of database (s669/fireprot), \
                        assuming you are in the root of the repository',
                      default='fireprot')
    parser.add_argument('--db_loc', help='location of database,\
                        only specify if you are using a custom DB',
                       default=None)
    parser.add_argument('-o', '--output_root', 
                        help='root of folder to store outputs',
                        default='.')
    parser.add_argument('-i', '--internal_path', 
                        help='modified path to outputs at inference computer',
                        default=None)
    parser.add_argument('-a', '--alignments',
                        help='folder where redundancy-reduced alignments are',
                        default='./data/msas')
    parser.add_argument('--rosetta', action='store_true', 
        help='whether to get Rosetta offsets'
            +' (only use when Rosetta relax has been run')
    parser.add_argument('--inverse', action='store_true', 
        help='whether to get offsets from (Robetta) predicted mutant structures'
            +' only use when Rosetta relax has been run on Robetta structures')
    parser.add_argument('--verbose', action='store_true',
        help='whether to save which mutations could not be parsed')
    parser.add_argument('--infer_pos', action='store_true')

    args = parser.parse_args()
    if args.dataset.lower() in ['fireprot', 'fireprotdb']:
        args.db_loc = './data/fireprotdb_results.csv'
    elif args.dataset.lower() in ['s669', 's461']:
        args.db_loc = './data/Data_s669_with_predictions.csv'
        args.inverse = True
    else:
        print('Inferred use of user-created database. Note: this must '
              'contain columns for code, wild_type, position, mutation. '
              'position must correspond to PDB sequence')
        assert args.dataset != 'fireprot'
        assert args.db_loc is not None

    main(args)