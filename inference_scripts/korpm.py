import pandas as pd
import os
import sys
import argparse

def main(args):
    db = pd.read_csv(args.db_location)

    if 'fireprot' in args.db_location.lower():
        dataset = 'fireprot'
        db = db.groupby('uid').first()
        # mutation is defined by sequence, but we are using structure
        db['korpm_mut'] = db['wild_type'] + db['chain'] + \
            (db['position']+db['offset_up']).astype(str) + db['mutation']
        db['korpm_struct'] = db['code'] + '_' + db['chain']
        db = db[['korpm_struct', 'korpm_mut']]
        #print(db[db['korpm_mut'].str.contains('LA196')])
        # single mutation which KORPM cannot handle
        db = db.loc[~(db['korpm_mut'].str.contains('LA196'))]

    elif 's669' in args.db_location.lower():
        dataset = 's669'
        # mutation is defined using structure

        if not args.inverse:
            db['korpm_mut'] = db['wild_type'] + db['chain'] + \
                db['position'].astype(str) + db['mutation']
            db['korpm_struct'] = db['code'] + '_' + db['chain']
            db = db[['korpm_struct', 'korpm_mut']]
            #print(db[db['korpm_mut'].str.contains('RA243E')])
            # mutations which KORPM cannot handle
            db = db.loc[~(db['korpm_mut'].str.contains('CA67'))]
            db = db.loc[~(db['korpm_mut'].str.contains('RA243E'))]

        else:
            dataset += '_inv'
            # manual reassignments since Robetta structures had chain names
            # reassigned; incompatible with previous labelling
            db.loc[db['code']=='1IV7', 'chain'] = 'B'
            db.loc[db['code']=='3L15', 'chain'] = 'B'
            db.loc[db['code']=='3L15', 'position'] = 200
            db.loc[db['code']=='4N6V', 'chain'] = '0'
            db['korpm_mut'] = db['mutation'] + 'A' + \
                db['position'].astype(str) + db['wild_type']
            db['korpm_struct'] = db['code'].str.lower() + db['chain'] + '_' + \
                db['wild_type'] + db['position'].astype(str) + db['mutation']

    else:
        print('Database name must contain either "db" or "db"')
        exit()

    db.to_csv(
        os.path.join(args.korpm_dir, f'{dataset}_korpm.csv'), 
        header=False, index=False, sep=' '
        )

    korpm_input = dataset+'_korpm.csv'
    korpm_output = dataset+'_korpm_preds.txt'

    cmd = f"{os.path.join(args.korpm_dir, 'sbg', 'bin', 'korpm')}" \
          f" {os.path.join(args.korpm_dir, korpm_input)}" \
          f" --dir {args.structures_dir} --score_file " \
          f" {os.path.join(args.korpm_dir, 'pot', 'korp6Dv1.bin')}" \
          f" -o {os.path.join(args.korpm_dir, korpm_output)}" 

    print(cmd)
    os.system(cmd)

    korpm_preds = pd.read_csv(
        os.path.join(args.korpm_dir, f'{dataset}_korpm_preds.txt'), 
        sep='\s+', header=None
        )
    korpm_preds.columns = [
        'korpm_struct', 'korpm_mut', 
        f'korpm{"_dir" if "inv" not in dataset else "_inv"}'
        ]
    korpm_preds = db.reset_index().merge(
        korpm_preds, on=['korpm_mut', 'korpm_struct']
        ).set_index('uid').drop_duplicates()

    preds_db = pd.read_csv(args.output, index_col=0)
    preds_db = preds_db.join(
        korpm_preds[f'korpm{"_dir" if "inv" not in dataset else "_inv"}']
        )
    preds_db.to_csv(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Score sequences based on a given structure.'
    )
    parser.add_argument(
            '--korpm_loc', type=str,
            help='location of the korpm installation',
    )
    parser.add_argument(
            '--structures_dir', type=str,
            help='location of preprocessed structures from preprocess.py'
    )
    parser.add_argument(
            '--db_location', type=str,
            help='location of the mapped database (db/db)_mapped.csv',
    )
    parser.add_argument(
            '--output', '-o', type=str,
            help='location where the predictions will be stored, which should\
                be a copy of the mapped database with additional cols'
    )
    parser.add_argument(
            '--inverse', action='store_true', default=False,
            help='use the mutant structure and apply a reversion mutation'
    )

    args = parser.parse_args()
    main(args)
