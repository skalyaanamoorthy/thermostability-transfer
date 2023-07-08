import urllib
import os
import gzip
import requests
import re
import shutil

from Bio import pairwise2
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1

from modeller import *
from modeller.scripts import complete_pdb

import pandas as pd


def download_assembly(code, chain, BIO_ASSEMBLIES_DIR):
    """
    Downloads the full (multimeric) biological assembly associated with a given
    PDB code as a gzip file.
    """
        
    print(f"Parsing {code}")

    # don't re-download (time-consuming)
    prot_file = f'{code}_{chain}.pdb1.gz'
    prot_path = os.path.join(BIO_ASSEMBLIES_DIR, prot_file)    
    if not os.path.exists(prot_path):
        print(f'Downloading {code} biological assembly')
        try:
            # for some reason this is the one bio assembly that doesn't exist
            # so use monomer instead
            if code == '1W4E':
                urllib.request.urlretrieve(
                    f'http://files.rcsb.org/download/{code}.pdb.gz',
                    prot_path
                )
            else:
                urllib.request.urlretrieve(
                    (f'https://files.wwpdb.org/pub/pdb/data/biounit/PDB/all/'
                    f'{code.lower()}.pdb1.gz'), 
                    prot_path
                )
        # only happens due to connection issues
        except Exception as e:
            print(e)
            print(f'Downloading {code} failed')
    
    # convert the file with a different structure
    if code == '1W4E':
        lines = gzip.open(prot_path, 'rt').readlines()
        lines = lines[:954]
        with gzip.open(prot_path, 'wt') as g:
            g.writelines(lines)

    return prot_path, prot_file


def get_uniprot_s669(code, chain, SEQUENCES_DIR):
    """
    Gets the UniProt sequence (and accession code, for computing features)
    for specifically S669 / S461 proteins, since these are not provided with
    the database. Uses the PDB sequence if this cannot be found.
    """

    # uniprot entries corresponding to multichain PDBs may need to be specified   
    if code == '1GUA' or code == '1GLU' or code == '2CLR' or code == '3MON':
         entity = 2
    elif code == '3DV0' or code == '1HCQ':
         entity = 3
    else:
         entity = 1

    # get the uniprotkb data associated with the PDB code if it exists
    req = (
        f'https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/uniprot_mapping/'
        f'{code.lower()}/{entity}'
    )

    # convert json to Python
    r = requests.get(req).text.replace('true','True').replace('false','False')
    r = eval(r)

    # get specifically the sequence related to the target structure
    try:
        data = r[code.lower()]
        # get the uniprotkb accession (skip interpro entries which have _)
        num = -1
        accession = '_'
        while '_' in accession:
            num += 1
            accession = data['data'][num]['accession']

        # query uniprotkb for the accession to get the FULL sequence 
        # (used for alignment searching as it gives improved performance)
        req2 = f'https://rest.uniprot.org/uniprotkb/{accession}'
        r2 = requests.get(req2).text
        r2 = r2.split('"sequence":{"value":')[-1].split(',')[0].strip('\""')
        uniprot_seq = r2

        with open(
            os.path.join(SEQUENCES_DIR, 'fasta_up', f'{code}_{chain}.fa'), 'w'
            ) as f:
                f.write(f'>{code}_{chain}\n{uniprot_seq}')
               
    # e.g. 1FH5, which is an FAB fragment
    except KeyError:
        print(f'No UP for {code}')
        uniprot_seq = None
        accession = None

    # UniProt sequence has incorrect residues for second half of protein
    # so just use the PDB sequence for searching
    if code in ['1IV7', '1IV9']:
        uniprot_seq = None

    return uniprot_seq, accession


def renumber_pdb(pdb_file, output_file):
    """
    Renumbers the residues in the PDB file sequentially
    """

    # Parse the structure
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file)

    # Temporarily renumber the residues with a large offset to avoid conflicts
    # (where two residues share the same identity)
    offset = 10000
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain.get_list(), start=1):
                residue.id = (' ', i + offset, ' ')

    # Sequentially renumber the residues (starting from 1)
    for model in structure:
        for chain in model:
            residues = sorted(chain.get_list(), key=lambda res: res.get_id()[1])
            for i, residue in enumerate(residues, start=1):
                residue.id = (' ', i, ' ')

    # Write the output file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)


def repair_pdb(pdb_file, output_file):
    """
    Repairs missing atoms / residues using Modeller. Requires Modeller and
    the LIB environment variable to be set to the appropriate directory
    """

    # setup Modeller
    env = Environ()
    env.libs.topology.read(file='$(LIB)top_heav.lib')
    env.libs.parameters.read(file='$(LIB)par.lib')
    # get the PDB code
    filename = os.path.basename(pdb_file)

    # repairing these structures causes as numbering conflict with UniProt
    if filename[:4] in ['1G3P', '1IR3', '4HE7']:
        return
    # the other chain in these structures is DNA, causing errors
    if filename[:4] in ['1AZP', '1C8C']:
        mdl = complete_pdb(env, pdb_file, model_segment=('1:A', 'LAST:A'))
    # usually nothing is missing, so the structure is unchanged
    else:
        mdl = complete_pdb(env, pdb_file)
    mdl.write(file=output_file)


def extract_structure(code, chain, d, prot_path, prot_file, STRUCTURES_DIR):
    """
    Using the gzip assembly from the PDB, parse the file to get the sequence of 
    interest from the structure
    """

    pdbparser = PDBParser()

    # the chains will end up getting renamed, as sometimes in the assembly
    # two chains will share a name, causing errors
    chain_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
    lines = gzip.open(prot_path, 'rt').readlines()

    # chain whose sequence we want (becomes mutated)
    target_chain = chain
    target_chain_found = False
    new_idx = -1
    skip_ter = False
    skip_hets = False
     
    # this section puts all structures in a consistent format
    # first, specify the output location used for inference
    target_structure = re.sub(
        '.pdb1.gz', '.pdb', os.path.join(STRUCTURES_DIR, prot_file)
        )
    with open(target_structure, 'w') as f:
        # lines from gzip
        for line in lines:
            # erase model lines / replace with TER and replace chain as needed 
            # unless it is target. partly to ensure chain name uniqueness

            # multiple models usually implies NMR structures, but all bio-
            # assemblies have at least one model
            if line[0:5] == 'MODEL':
                new_idx += 1
                # select a new unique chain name
                chn = chain_names[new_idx]
                # if this is the next model, terminate the previous chain
                if int(new_idx) > 0 and not skip_ter:
                    f.write('TER\n')
                elif skip_ter:
                    skip_ter = False
                if chn == target_chain:
                    new_idx += 1
                    chn = chain_names[new_idx]
                continue
            # don't output this
            elif line[0:6] == 'ENDMDL':
                continue
            # rewrite lines with atom records according to the new designation
            elif line[0:4] == 'ATOM':
                # by default, include heteroatoms which occur within a chain
                # since they are probably associated with usual residues
                skip_hets = False
                # reassign the chain, unless it is the target chain, in which
                # case there are checks to ensure the designation is correct
                if line[21]==target_chain and not target_chain_found:
                    target_chain_found = True
                    chn = target_chain
                f.write(line[:21]+chn+line[22:])
                continue
            # remove heteroatoms which are not presumed residues 
            # (usually ligands and waters)
            elif line[0:6] == 'HETATM' and skip_hets:
                continue
            elif line[0:6] == 'HETATM':
                # it is possible for the target chain to start with HETS
                if line[21]==target_chain and not target_chain_found:
                    target_chain_found = True
                    chn = target_chain
                if line[17:20] not in d.keys():
                    # original residue      
                    old = line[17:20]
                    line = list(line)
                    # exclude sepharose, aminosuccinimide, acetyl
                    if ''.join(line[17:20]) in ['SEP', 'SNN', 'ACE']:
                        print(
                            f'Omitting residue {"".join(line[17:20])} in {code}'
                            )
                        continue
                    else:
                        # ESM-IF uses a 3-1 letter encoding that cannot handle 
                        # nonstandard residues except 'UNK'
                        line[17:20] = 'UNK'
                    line = ''.join(line)
                    print(f'Converted {old} in {code} to {line[17:20]}')
                f.write(line[:21]+chn+line[22:])
                continue               
            elif line[0:3] == 'TER':
                f.write(line[:21]+chn+line[22:])
                new_idx += 1
                # when moving on to a new chain, chose a new name which is not 
                # the name of the target
                chn = chain_names[new_idx]
                if chn == target_chain:
                    new_idx += 1
                    chn = chain_names[new_idx]
                # deletes waters and ligands by default
                skip_hets = True
                # ensures additional TER won't be written
                skip_ter = True
                continue
            f.write(line)

    # repair missing atoms / residues
    repair_pdb(target_structure, target_structure)
    # renumber sequentially
    renumber_pdb(target_structure, target_structure)

    # add in a CRYST1 line so that DSSP will accept the file
    text_to_insert = \
     'CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1 '

    with open(target_structure, 'r') as original_file:
        lines = original_file.readlines()

    if 'MODELLER' in lines[0]:
        lines.insert(1, text_to_insert + '\n')
    else:
        lines.insert(0, text_to_insert + '\n')

    with open(target_structure, 'w') as modified_file:
        modified_file.writelines(lines)

    # the remainder can be handled by an existing parser
    structure = pdbparser.get_structure(code, target_structure)

    # create a mapping like: (chain) A: [(1, MET), (2, PHE), ...]
    chains = {chain.id:[(residue.id[1], residue.resname) for residue in chain]\
        for chain in structure.get_chains()}

    return chains


def align_sequence_structure(code, chain, pdb_seq, dataset, d,
                             SEQUENCES_DIR, WINDOWS_DIR, ALIGNMENTS_DIR, 
                             uniprot_seq=None):
    """
    In this critical preprocessing step, the mutations from the database are 
    mapped to the structures that will be used by inverse folding methods 
    downstream. Since the predictive methods can be finicky about data format, 
    some manual corrections are made. This function also writes the alignments 
    for validation, and selects a window of the full UniProt sequence which best 
    encompasses the structure while maximimizing length.
    """

    # use previous mapping to generate a 1-character code for each residue
    pdb_ungapped = ''.join([d[res[1]] for res in pdb_seq])

    # hand-adjusted alignments
    if code == '1LVM':
        pdb_gapped = ['-']*2029 + list(pdb_ungapped)
        pdb_gapped = ''.join(pdb_gapped)
        uniprot_gapped = uniprot_seq
    elif code == '1GLU':
        pdb_gapped = ['-']*433 + list(pdb_ungapped)
        pdb_gapped = ''.join(pdb_gapped)
        uniprot_gapped = uniprot_seq
    elif code == '1TIT':
        pdb_gapped = ['-']*12676 + list(pdb_ungapped)
        pdb_gapped = ''.join(pdb_gapped)
        uniprot_gapped = uniprot_seq
    #elif code == '1JNX':
    #    pdb_gapped = pairwise2.align.globalms(
    #       uniprot_seq, pdb_ungapped, 2, 0.5, -1, -0.1)[0].seqB 
    #    uniprot_gapped = pairwise2.align.globalms(
    #       pdb_gapped, uniprot_seq, 2, 0.5, -1, -0.1)[0].seqB

    # in most cases, it suffices to do an automatic alignment
    elif uniprot_seq is None:
        pdb_gapped = pdb_ungapped
        uniprot_gapped = pdb_ungapped
        uniprot_seq = pdb_ungapped
        # create a "fake" UniProt sequence from the PDB sequence if 
        # no UniProt is associated
        fake_up_seq = pdb_ungapped
        # would normally be saved when it is found 
        with open(
            os.path.join(SEQUENCES_DIR, 'fasta_up', f'{code}_{chain}.fa'), 
            'w') as f:
            f.write(f'>{code}_{chain}\n{fake_up_seq}')
    else:
        # get the highest-scoring candidate alignment
        aln = pairwise2.align.globalms(
            uniprot_seq, pdb_ungapped, 2, 0.5, -1, -0.1
            )[0]
        uniprot_gapped = aln.seqA
        # pdb_gapped is the PDB sequence, with added gaps (-) to match up 
        # with  FireProtDB (UniProt) sequences.
        pdb_gapped = aln.seqB

    # hand-adjusted alignment
    if code == '1AAR':
        pdb_gapped = ['-']*608 + list(pdb_ungapped)
        pdb_gapped = ''.join(pdb_gapped)

    #elif dataset == 's669':

    #    # automatical alignments
    #    elif uniprot_seq is not None:
    #        aln = pairwise2.align.globalms(
    #            uniprot_seq, pdb_ungapped, 2, 0.5, -1, -0.1
    #            )[0]
    #        uniprot_gapped = aln.seqA
    #        pdb_gapped = aln.seqB
    #    # use the PDB sequence if there is no UniProt sequence found
    #    else:


    # dataframe which shows how the sequence-structure alignment was created
    alignment_df = pd.DataFrame(index=range(100000))
    # paste in the sequences
    alignment_df['uniprot_gapped'] = pd.Series(list(uniprot_gapped))
    alignment_df['pdb_gapped'] = pd.Series(list(pdb_gapped))
    # drop extra rows
    alignment_df = alignment_df.dropna()

    # for the MSA transformer, we can only have 1022 characters in the sequence
    # this block extracts the most relevant residues up to 1022 from the UniProt
    # sequence: the first 1022 residues which alsoully cover the structure, 
    # extending past the N- and then C- terminus if there is space

    # the alignment process often 'floats' an M far away from the rest of the
    # sequence. This is inconsequential, except for this line where we use the
    # coordinate of the second aligned residue and subtract one
    window_start = alignment_df.loc[
        alignment_df['pdb_gapped']!='-'].head(2).tail(1).index.item() - 1
    window_end = alignment_df.loc[
        alignment_df['pdb_gapped']!='-'].tail(1).index.item()
    # we now have the start and end of the UniProt's overlap with the structure
    # so now we extend it up to 1022
    while window_start > 0 and window_end - window_start < 1022:
        window_start -= 1
    # ran out of sequence at N-term, so add some on the C-term
    while window_end < len(alignment_df) and window_end - window_start < 1022:
        window_end += 1
    # this is the sequence used by MSA-Transformer and ESM-1V
    # but not Tranception
    uniprot_seq_reduced = uniprot_seq[window_start:window_end]
    with open(os.path.join(WINDOWS_DIR, f'{code}_{chain}'), 'w') as f:
        f.write(f'{window_start},{window_end}')
    with open(os.path.join(WINDOWS_DIR, f'{code}_{chain}.fa'), 'w') as f:
        f.write(f'>{code}_{chain}\n{uniprot_seq_reduced}')
    alignment_df.to_csv(
        os.path.join(ALIGNMENTS_DIR, f'{code}_uniprot_aligned.csv')
        )

    return alignment_df, window_start, pdb_ungapped, uniprot_seq


def get_offsets(wt, pos, dataset, alignment_df):
    """
    Count the number of gaps in the uniprot sequences (caused by insertions in 
    the PDB structure), ensuring that all gaps prior to the target mutation 
    position are counted is preserved
    """

    # location of the mutation in the alignment
    # indexing is based on UniProt for FireProtDB, PDB for S669
    if dataset == 'fireprot':
        idx_mut = alignment_df.loc[
            alignment_df['uniprot_gapped']!='-'].head(pos).tail(1).index.item()
    elif dataset == 's669':
        idx_mut = alignment_df.loc[
            alignment_df['pdb_gapped']!='-'].head(pos).tail(1).index.item()
    
    # case where the PDB is mutated relative to UniProt
    if alignment_df.loc[idx_mut, 'pdb_gapped'] != wt:
        print(f'Could not match wild-type residue {wt} to position {pos}')

    try:
        pdb_gaps = alignment_df.loc[
            :idx_mut, 'pdb_gapped'].value_counts()['-'] 
    # case where there is no gap
    except KeyError:
        pdb_gaps = 0
    try:
        uniprot_gaps = alignment_df.loc[
            :idx_mut, 'uniprot_gapped'].value_counts()['-']
    except KeyError:
        uniprot_gaps = 0

    offset_up = uniprot_gaps - pdb_gaps

    return offset_up


def generate_mutant_sequences(code, chain, pos, mut, pdb_ungapped, offset_up, 
                              SEQUENCES_DIR):
    """
    Save the sequences of the wild-type and mutant proteins, returning the 
    latter. Mainly for record-keeping.
    """

    with open(
        os.path.join(SEQUENCES_DIR, 'fasta_wt', f'{code}_{chain}_PDB.fa'), 
        'w') as f:
        f.write(f'>{code}_{chain}\n{pdb_ungapped}')

    # modify the string in the target position
    mutseq = list(pdb_ungapped)
    mutseq[int(pos) - 1 + offset_up] = mut
    mutseq = ''.join(mutseq)

    with open(
        os.path.join(SEQUENCES_DIR, 'fasta_mut', f'{code}_{chain}_PDB.fa'),
        'w') as f:
        f.write(f'>{code}_{chain}\n{mutseq}')

    return mutseq


def save_formatted_data(code, chain, group, dataset, output_root):
    """
    Save information about the mutants in the formats expected for each method.
    """

    # open the Tranception file
    with open(os.path.join(
        output_root, 'DMS_tranception', f'{code}_{chain}_{dataset}.csv'
        ), 'w') as trance:
        trance.write('mutated_sequence,mutant\n')

        # Open the MSA-Transformer file (also used by ESM-1V)
        with open(os.path.join(
            output_root, 'DMS_MSA', f'{code}_{chain}_{dataset}.csv'),
             'w') as msa:
            msa.write(',mutation\n')

            # iterate through the mutations, writing each one after validation
            for (wt, pos, mut, ou, seq, ws), _ in group.groupby(
                ['wild_type', 'position', 'mutation', 
                'offset_up', 'uniprot_seq', 'window_start']):

                uniprot_seq = list(seq)
                try:
                    assert uniprot_seq[
                        pos - 1 + ou * (-1 if dataset=="s669" else 0)] == wt,\
                        ('Wrote a mutation whose wt disagrees with uniprot_seq')
                except Exception as e:
                    print(e, code, wt, pos, mut)

                # generation of the mutant sequence
                uniprot_seq[pos - 1 + ou * (-1 if dataset=="s669" else 0)] = mut
                mutated_uniprot_seq = ''.join(uniprot_seq)

                # write to the Tranception file
                trance.write(f'{mutated_uniprot_seq},'
                             f'{wt}{pos + ou * (-1 if dataset=="s669" else 0)}'
                             f'{mut}\n')
                
                # edge case where structure is too large (happens once)
                if (pos + ou * (-1 if dataset=="s669" else 0) - ws > 1022):
                    print('mutation occurs outside required window:', 
                            code, wt, pos, mut)
                else:
                    new_pos = pos + ou * (-1 if dataset=="s669" else 0) - ws
                    msa.write(f',{wt}{new_pos}{mut}\n')


def get_rosetta_mapping(code, chain, group, dataset, 
                        SEQUENCES_DIR, ROSETTA_DIR, inverse=False):
    """
    Find additional offsets caused by Rosetta failing to handle any residues.
    Can only be used after generating relaxed structures.
    """

    # inv(erse) refers to the predicted mutant structures for reversions in S669
    colname = 'offset_inv' if inverse else 'offset_rosetta' 
    offsets_rosetta = pd.DataFrame(
        columns=['code', 'chain', 'position', 'mutation', colname]
        )

    for (wt, pos, mut, ou, orig_seq, path), _ in group.groupby(
            ['wild_type', 'position', 'mutation', 
             'offset_up', 'pdb_ungapped', 'mutant_pdb_file']):

        # indexing is already based on structure, UniProt offset is irrelevant
        if dataset == 's669':
            ou = 0 

        rosparser = PDBParser(QUIET=True)
        
        if inverse:
            # get only the file name, not the path
            fname = path.split('/')[-1]
            full_path = os.path.join(ROSETTA_DIR, 
                f'{code}_{chain}', f'{code}_{pos}{mut}', 'inv_robetta', fname)
            # assume the minimized structure is generated and is located here
            ros = re.sub('.pdb', '_inv_minimized_0001.pdb', full_path)
        else:
            full_path = os.path.join(ROSETTA_DIR, 
                f'{code}_{chain}', f'{code}_{chain}.pdb')
            # assume the minimized structure is generated and is located here
            ros = re.sub('.pdb', '_minimized_0001.pdb', full_path)

        # try to obtain the sequence from the minimized structure
        try:
            structure = rosparser.get_structure(code, ros)
            chains = {c.id:seq1(''.join(residue.resname for residue in c)) 
                    for c in structure.get_chains()}
            ros_seq = chains[chain if not inverse else 'A']
        except Exception as e:
            print(e)
            print(f'Structure or chain not found: {code} {chain} {ros}')
            return None
    
        with open(os.path.join(
                SEQUENCES_DIR, 
                f"fasta_wt/{code}_{chain}_rosetta{'_inv' if inverse else ''}.fa"
            ),'w') as f:
            f.write(f'>{code}_{chain}\n{ros_seq}')

        # reconstruct the wild-type sequence to match the structure
        if not inverse:
            orig_seq = list(orig_seq)
            orig_seq[int(pos)-1+ou] = wt
            orig_seq = ''.join(orig_seq)

        # align the rosetta sequence to the original pdb sequence to see if 
        # anything was deleted
        aln = pairwise2.align.globalms(
            orig_seq, ros_seq, 5, 0, -2, -0.1)[0] #match, miss, open, extend
        ros_gapped = aln.seqB
        
        #offset_rosetta: offset due to residues dropped by rosetta
        offset_rosetta = ros_gapped[:int(pos) + ou].count('-')

        if offset_rosetta != 0:
            print(f'Rosetta removed {offset_rosetta} residues: {code}')

        # validate the mutation (the inverse uses the mutant structure)
        try:
            if inverse:
                assert ros_seq[int(pos) - 1 + ou - offset_rosetta] == mut
            else:
                assert ros_seq[int(pos) - 1 + ou - offset_rosetta] == wt

        # sometimes rosetta deletes a mutated residue
        except:
            print('Deleted mutant residue:', 
                code, wt, pos, mut, '\n', orig_seq, '\n', gapped)

        offsets_rosetta = pd.concat([offsets_rosetta, pd.DataFrame({
                    0:{'code':code, 'chain':chain, 'position':pos, 
                       'mutation':mut, colname:offset_rosetta}
                    }).T])
                
    return offsets_rosetta


def create_db_from_mutseq(code, pdb_seq, mut_seq, output_dir):
    """
    Used for multimutants only.
    """
    
    alignment = pairwise2.align.globalms(pdb_seq, mut_seq, 2, 0.5, -1, -0.1)[0]

    data = []
    index_in_seqA = 1  # Assuming 1-based index for positions

    for wt, mut in zip(alignment.seqA, alignment.seqB):
        if wt != mut:
            data.append({'position': index_in_seqA, 'wild_type': wt, 'mutation': mut})
        index_in_seqA += 1

    df = pd.DataFrame(data)
    df['uid'] = code + '_' + df['position'].astype(str) + df['mutation']
    df.to_csv(os.path.join(output_dir, 'muts.csv'))
    return df