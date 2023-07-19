import os
import pandas as pd
import numpy as np
import glob
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import matplotlib
import re
from tqdm.notebook import tqdm
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from scipy import stats
from seaborn import diverging_palette
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.lines import Line2D


# convert names used in inference outputs to those used in figures
remap_names = {
    'esmif_monomer': 'ESM-IF(M)', 
    'esmif_multimer': 'ESM-IF', 
    'esmif_multimer_masked': 'ESM-IF(masked)',
    'mpnn_mean': 'ProteinMPNN_mean', 
    'msa_transformer_mean': 'MSA-T_mean', 
    'esm1v_mean': 'ESM-1V_mean',
    'mif': 'MIF', 
    'mifst': 'MIF-ST', 
    'monomer_ddg': 'Ros_ddG_monomer', 
    'cartesian_ddg': 'Rosetta_CartDDG', 
    'mpnn_10_00': 'ProteinMPNN_10', 
    'mpnn_20_00': 'ProteinMPNN_20', 
    'mpnn_30_00': 'ProteinMPNN_30', 
    'tranception': 'Tranception',    
    'esm1v_2': 'ESM-1V_2', 
    'msa_1': 'MSA-T_1', 
    'korpm': 'KORPM', 
    'msa_transformer_median': 'MSA-T_median'
    }

# predictions will have dir in their name to specify direct mutation
remap_names_2 = {f"{key}_dir": value for key, value in remap_names.items()}
remap_names_2.update({
    'ddG_dir': 'ΔΔG label', 
    'dTm_dir': 'ΔTm label', 
    'random_dir': 'Gaussian noise', 
    'delta_kdh_dir': 'Δ hydrophobicity', 
    'delta_vol_dir': 'Δ volume', 
    'abs_ASA_dir': 'SASA'
    })


def parse_cartesian(filename, reduce='mean'):
    # read the file into a dataframe
    df = pd.read_csv(filename, delim_whitespace=True, header=None)
    # we used 3 conformational samples for both the wild-type and mutant
    # taking the lowest-energy structure generally leads to worse results
    
    # take the average of the fourth field for the first 3 lines
    if reduce == 'mean':
        reduced = df.groupby(2)[3].mean()
    elif reduce == 'min':
        reduced = df.groupby(2)[3].min()

    # group means/mins
    wt_red = reduced.loc['WT:']
    mut_red = reduced.drop('WT:').item()

    return float(wt_red - mut_red)


def parse_rosetta_predictions(df, root_name, inverse=False, reduce='mean', kind='cartesian'):
    """
    loads results from Rosetta, which are included in the repo
    """
    df_rosetta = pd.DataFrame(columns=[f"cartesian_ddg{'_inv' if inverse else '_dir'}"] if kind=='cartesian' else [f"monomer_ddg{'_inv' if inverse else '_dir'}"])
    df_rosetta_runtimes = pd.DataFrame(columns=['runtime_'+df_rosetta.columns[0]])
    missed = []

    # including in the repo are predictions and runtimes associated with each uid
    for uid in sorted(df.reset_index()['uid'].unique()):
        
        if not inverse:
            pred_path = os.path.join(root_name, uid + '.ddg')
            rt_path = os.path.join(root_name, 'runtime_' + uid + '.txt')
        else:
            pred_path = os.path.join(root_name, uid + '_inv.ddg')
            rt_path = os.path.join(root_name, 'runtime_' + uid + '_inv.txt')
        
        if not os.path.exists(pred_path):
            print('Could not find predictions for', uid)

        df_rosetta.at[uid, f"cartesian_ddg_{'inv' if inverse else 'dir'}"] = parse_cartesian(pred_path, reduce=reduce)
        df_rosetta_runtimes.at[uid, f"runtime_cartesian_ddg_{'inv' if inverse else 'dir'}"] = int(open(rt_path, 'r').read().strip())          

    return df_rosetta, df_rosetta_runtimes


def compute_ndcg(dbf, pred_col, true_col):
    # Shift scores to be non-negative
    df = dbf.copy(deep=True)
    min_score = df[true_col].min()
    shift = 0
    if min_score < 0:
        shift = -min_score

    df[true_col] += shift
    
    # Sort dataframe by ground truth labels
    df_sorted = df.sort_values(by=pred_col, ascending=False)
    
    # Get the sorted predictions
    sorted_preds = df_sorted[pred_col].values
    
    # Use the ground truth labels as relevance scores
    relevance = df_sorted[true_col].values
    
    # Reshape data as ndcg_score expects a 2D array
    sorted_preds = sorted_preds.reshape(1, -1)
    relevance = relevance.reshape(1, -1)
    
    # Compute and return NDCG
    try:
        return metrics.ndcg_score(relevance, sorted_preds)
    except:
        print(pred_col)
        print(sorted_preds)
        print(true_col)
        print(relevance)


def recovery_curves(rcv, models=['cartesian_ddg_dir', 'ddG_dir', 'dTm_dir', 'random_dir'], spacing=0.02):

    def annotate_points(ax, data, x_col, y_col, hue_col, x_values, text_offset=(0, 0), spacing=0.02):
        line_colors = {}
        for line in ax.lines:
            label = line.get_label()
            color = line.get_color()
            line_colors[label] = color

        for x_val in x_values:
            models_and_points = []
            for model, model_data in data.groupby(hue_col):
                value_row = model_data.loc[model_data[x_col] == x_val]
                if not value_row.empty:
                    x, y = value_row[x_col].values[0], value_row[y_col].values[0]
                    models_and_points.append((model, x, y))

            # Sort models_and_points by y values to space them evenly
            models_and_points = sorted(models_and_points, key=lambda x: x[2], reverse=True)

            # Calculate annotation positions and add annotations
            y_annot = max(y for _, _, y in models_and_points) - text_offset[1]
            for model, x, y in models_and_points:
                ax.annotate(f"{y:.2f}", (x, y),
                            xytext=(x - text_offset[0], y_annot),
                            arrowprops=dict(arrowstyle='-', lw=1, color='gray'),
                            fontsize=9, color=line_colors[model])
                y_annot -= spacing
                ax.axvline(x=x, color='r', linestyle='dashed')

    font = {'size'   : 12}
    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    measurement = 'ddG'

    d5 = rcv.reset_index()
    d5 = d5.loc[d5['model'].isin(models)].set_index(['measurement', 'model_type', 'model', 'class'])
    d5 = d5.drop([c for c in d5.columns if 'stab_' in c], axis=1)
    # for plotting recovery over thresholds

    recov = d5[[c for c in d5.columns if '%' in c]].reset_index()
    recov = recov.loc[recov['model']!='dTm_dir']
    #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
    recov = recov.loc[recov['measurement']==measurement]
    recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
    melted_1 = recov.melt(id_vars='model', value_vars=[str(int(s))+'%' for s in range(101)], var_name="variable", value_name="value")
    #melted_2 = recov.melt(id_vars='model', value_vars=['pos_'+str(int(s))+'$' for s in range(101)],var_name="pos_variable", value_name="pos")
    #melted_1['suffix'] = melted_1['variable']
    #melted_2['suffix'] = melted_2['pos_variable'].str[4:]
    #recov = pd.merge(melted_1, melted_2, on=['model', 'suffix']).drop(columns=["pos_variable", "suffix"])
    recov = melted_1

    recov['variable'] = recov['variable'].str.strip('%').astype(float)
    sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=axes[0, 0])
    #pos = recov.groupby('variable').max().reset_index()
    #pos['pos'] /= pos['pos'].max()
    #sns.lineplot(data=pos, x='variable', y='pos', ax=axes[0,0])
    #axes[0, 0].set_xlabel('percentile of predictions and measurements compared')
    axes[0, 0].set_xlabel('percentile of predictions assessed')
    #axes[0, 1].set_ylabel('fraction of top mutants identified')
    axes[0, 0].set_ylabel('fraction stabilizing')
    axes[0, 0].set_title('ΔΔG')
    annotate_points(axes[0, 0], recov, 'variable', 'value', 'model', [90], text_offset=(20, -0.05), spacing=spacing)

    recov = d5[[c for c in d5.columns if 'x_recovery' in c]].reset_index()
    recov = recov.loc[recov['model']!='dTm_dir']
    #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
    recov = recov.loc[recov['measurement']==measurement]
    recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
    recov = recov.melt(id_vars='model')
    recov['variable'] = recov['variable'].str.strip('x_recovery').astype(float)
    sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=axes[1, 0])
    axes[1, 0].set_xlabel('multiple of each protein\'s stablizing mutants screened')
    axes[1, 0].set_ylabel('fraction recovered')
    annotate_points(axes[1, 0], recov, 'variable', 'value', 'model', [0.5], text_offset=(0.4, -0.2), spacing=spacing)
    annotate_points(axes[1, 0], recov, 'variable', 'value', 'model', [1], text_offset=(-0.4, 0.1), spacing=spacing+0.02)
    #axes[1, 0].set_xscale('log')
    #axes[1, 0].set_yscale('log')

    measurement = 'dTm'

    recov = d5[[c for c in d5.columns if '%' in c]].reset_index()
    recov = recov.loc[recov['model']!='ddG_dir']
    #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
    recov = recov.loc[recov['measurement']==measurement]
    recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
    recov = recov.melt(id_vars='model')
    recov['variable'] = recov['variable'].str.strip('%').astype(float)
    sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=axes[0, 1])
    #axes[0, 0].set_xlabel('percentile of predictions and measurements compared')
    axes[0, 1].set_xlabel('percentile of predictions assessed')
    #axes[0, 1].set_ylabel('fraction of top mutants identified')
    axes[0, 1].set_ylabel(None)
    axes[0, 1].set_title('ΔTm') #measurement_ = {'ddG': 'ΔΔG', 'dTm': 'ΔTm'}[measurement]
    annotate_points(axes[0, 1], recov, 'variable', 'value', 'model', [90], text_offset=(20, -0.05), spacing=spacing)

    recov = d5[[c for c in d5.columns if 'x_recovery' in c]].reset_index()
    recov = recov.loc[recov['model']!='ddG_dir']
    #recov = recov.loc[recov['model'].isin(['cartesian_ddg_dir', 'mpnn_20_00_dir', 'msa_transformer_mean_dir', 'esm1v_mean_dir'])]
    recov = recov.loc[recov['measurement']==measurement]
    recov = recov.drop(['measurement', 'model_type', 'class'], axis=1)
    recov = recov.melt(id_vars='model')
    recov['variable'] = recov['variable'].str.strip('x_recovery').astype(float)
    sns.lineplot(data=recov, x='variable', y='value', hue='model', ax=axes[1, 1])
    axes[1, 1].set_xlabel('multiple of each protein\'s stablizing mutants screened')
    #axes[1, 1].set_ylabel('fraction of stablizing mutants recovered')
    axes[1, 1].set_ylabel(None)
    annotate_points(axes[1, 1], recov, 'variable', 'value', 'model', [0.5], text_offset=(0.4, -0.2), spacing=spacing)
    annotate_points(axes[1, 1], recov, 'variable', 'value', 'model', [1], text_offset=(-0.4, 0.1), spacing=spacing+0.035)
    #axes[1, 1].set_xscale('log')
    #axes[1, 1].set_yscale('log')
    handles, labels = axes[0,0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.get_legend().remove()

    labels[labels.index('ddG_dir')] = 'Ground truth label'
    labels = [remap_names_2[name] if name in remap_names_2.keys() else name for name in labels]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.tight_layout()

    sp_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axes.flat):
        ax.text(-0.1, 1.1, sp_labels[i], transform=ax.transAxes, fontsize=16, va='top')

    plt.show()


def correlations(db_gt_preds, dbr, score_name, score_name_2=None, min_obs=5, bins=20, stat='spearman', runtime=False, 
                 group=True, valid=False, out=True, plot=False, coeff2=0.2, meas='ddG'):

    font = {'size'   : 14}
    matplotlib.rc('font', **font)
    
    # the first section is for testing combinations of models on the spot by making a custom score name
    df = db_gt_preds.copy(deep=True).reset_index()
    pattern = r"^(\w+) \+ (\w+) \* ([\d\.]+)$"
    # Use regex to match the pattern in the sample string
    match = re.match(pattern, score_name)
    
    if match and create:
        # Extract the parsed values from the regex match
        item_1 = match.group(1)
        item_2 = match.group(2)
        weight = float(match.group(3))
        combo = True

        assert item_1 in df.columns
        assert item_2 in df.columns
        if match.group(0) not in df.columns:
            df[score_name] = df[item_1] + df[item_2] * weight
            dbr[f'runtime_{score_name}'] = dbr[f'runtime_{item_1}'] + dbr[f'runtime_{item_2}']

    if score_name_2==None:
        score_name_2 = 'tmp'
        df[score_name_2]=0
        dbr[f'runtime_{score_name_2}']=0

    melted = df.melt(id_vars=['uid'], value_vars=['dTm', 'ddG']).dropna().rename({'variable':'type', 'value':'measurement'}, axis=1)
    df = melted.set_index('uid').join(df[['uid', 'code', 'ProTherm', score_name, score_name_2]].set_index('uid'))
    if runtime and score_name not in ['ddG_dir', 'dTm_dir', 'random_dir']:
        dbr = melted.set_index('uid').join(dbr[[f'runtime_{score_name}', f'runtime_{score_name_2}', 'code']])

    #todo: match colours between plots, add legends, add measurement distribution
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(score_name)

    g = df[['code', 'type', 'ProTherm', score_name, score_name_2, 'measurement']].dropna()
    g[f'{score_name} + {coeff2} * {score_name_2}'] = g[score_name] + coeff2 * g[score_name_2]

    if group:   
        if stat == 'spearman':
            i = pd.DataFrame()
            for (code, t, p), group in g.groupby(['code', 'type', 'ProTherm']):
                if len(group) > 1 and not all(group['measurement']==group['measurement'][0]):
                    ndcg1, _ = spearmanr(group['measurement'], group[score_name])
                    ndcg2, _ = spearmanr(group['measurement'], group[score_name_2])
                    tmp = pd.DataFrame([code, len(group), t, ndcg1, ndcg2, p]).T
                    i = pd.concat([i, tmp])
            i.columns=['code', 'obs', 'type', score_name, score_name_2, 'ProTherm']
            i = i.set_index(['code', 'obs', 'type'])
            ungrouped = pd.DataFrame()
            for t, group in g.groupby('type'):
                ug1, _ = spearmanr(group['measurement'], group[score_name])
                ug2, _ = spearmanr(group['measurement'], group[score_name_2])
                tmp = pd.DataFrame([len(group), t, ug1, ug2]).T
                ungrouped = pd.concat([ungrouped, tmp])
            ungrouped.columns=['obs', 'type', score_name, score_name_2]
            ungrouped = ungrouped.set_index('type')
        elif stat == 'ndcg':
            i = pd.DataFrame()
            for (code, t, p), group in g.groupby(['code', 'type', 'ProTherm']):
                if len(group) > 1 and not all(group['measurement']==group['measurement'][0]):
                    ndcg1 = compute_ndcg(group, score_name, 'measurement')
                    ndcg2 = compute_ndcg(group, score_name_2, 'measurement')
                    tmp = pd.DataFrame([code, len(group), t, ndcg1, ndcg2, p]).T
                    i = pd.concat([i, tmp])
            i.columns=['code', 'obs', 'type', score_name, score_name_2, 'ProTherm']
            i = i.set_index(['code', 'obs', 'type'])
            ungrouped = pd.DataFrame()
            for t, group in g.groupby('type'):
                ug1 = compute_ndcg(group, score_name, 'measurement')
                ug2 = compute_ndcg(group, score_name_2, 'measurement')
                tmp = pd.DataFrame([len(group), t, ug1, ug2]).T
                ungrouped = pd.concat([ungrouped, tmp])
            ungrouped.columns=['obs', 'type', score_name, score_name_2]
            ungrouped = ungrouped.set_index('type')

        if plot:
            axs[0,0].set_title(f'{stat} to ground truth')
            axs[0,1].set_title(f'distribution of predictions')
            sns.histplot(ax=axs[0, 1], x=score_name, data=g[[score_name, 'type']].reset_index(drop=True), alpha=0.3, hue='type', bins=bins)
            sns.histplot(ax=axs[1, 0], x='measurement', data=g[['measurement', 'type']].reset_index(drop=True), alpha=0.3, hue='type', bins=bins)
        if runtime and score_name not in ['ddG_dir', 'dTm_dir', 'random_dir']:
            runs = dbr[[f'runtime_{score_name}', f'runtime_{score_name_2}', 'code', 'type']].groupby(['code', 'type']).sum().reset_index()

    else:
        f = g.loc[g['type']==meas, [score_name, score_name_2, 'measurement']]
        n = len(f)
        i = f.corr('spearman')[['measurement', score_name]].drop('measurement').T
        i['obs'] = n
        if score_name_2 == 'tmp':
            f = f.drop('tmp', axis=1)
        if plot:
            sns.scatterplot(ax=axs[0,0], data=f, x=score_name, y=score_name_2, hue='measurement', alpha=0.3, palette='coolwarm_r')
            sns.scatterplot(ax=axs[0,1], data=f, x=score_name_2, y='measurement', hue=score_name, alpha=0.3, palette='coolwarm_r')
            sns.scatterplot(ax=axs[1,0], data=f, x=score_name, y='measurement', hue=score_name_2, alpha=0.3, palette='coolwarm_r')
            sns.scatterplot(ax=axs[1,1], data=f, x=f'{score_name} + {coeff2} * {score_name_2}', y='measurement', alpha=0.3, legend=None)
            plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
            
        return i
        
    if plot:
        data=i.reset_index()
        if stat == 'ndcg':
            data[score_name] = 100**data[score_name].astype(float)
            data[score_name_2] = 100**data[score_name_2].astype(float)
        
        sns.histplot(ax=axs[0,0], x=score_name, data=data[[score_name, 'type']].sort_values('type'), alpha=0.3, hue='type', bins=bins, stat='count', kde=True)
        axs[0,0].set_xlim((-1, 1))
        sns.scatterplot(ax=axs[1,1], data=g, x=score_name, y='measurement', hue='type', alpha=0.3)
        g = sns.jointplot(data=data, x=score_name_2, y=score_name, hue='type', kind='hist', marginal_kws=dict(bins=20), joint_kws=dict(alpha=0), height=10)
        for code, row in data.reset_index().iterrows():
            #if (row['obs'] > 50) and (row['code'] not in ('1RTB', '1BVC', '1RN1', '1BNI', '1BPI', '1HZ6', '1OTR', '2O9P', '1AJ3', '3VUB', '1LZ1')):# \
            if row['code'] in ['4E5K', '3D2A', '1ZNJ', '1WQ5', '1UHG', '1TUP', '1STN', '1QLP', '1PGA']:
                g.ax_joint.text(row[score_name_2]-0.01, row[score_name]-0.01, f"{row['code']}:{row['obs']}", size=8)
        ax = sns.scatterplot(data=data, x=score_name_2, y=score_name, hue='type', size='obs', style='ProTherm', sizes=(2,937), ax=g.ax_joint, alpha=0.4,
                            markers={True: "s", False: "o"})
        small = np.array(i[[score_name_2, score_name]].dropna()).min().min()
        big = np.array(data[[score_name_2, score_name]].dropna()).max().max()
        sns.lineplot(data=pd.DataFrame({'x': np.arange(small, big, 0.01), 'y': np.arange(small, big, 0.01)}), x='x', y='y', ax=g.ax_joint, color='red')
        plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)

        if stat == 'spearman':
            ax.set_xlabel(f'Spearman of {remap_names[ax.get_xlabel()[:-4]]}')
            ax.set_ylabel(f'Spearman of {remap_names[ax.get_ylabel()[:-4]]}')
        elif stat == 'ndcg':
            ax.set_xlabel(f'100^(NDCG) of {remap_names[ax.get_xlabel()[:-4]]}')
            ax.set_ylabel(f'100^(NDCG) of {remap_names[ax.get_ylabel()[:-4]]}')
        handles, labels = g.ax_joint.get_legend_handles_labels()
        legend = g.ax_joint.legend(handles, labels, loc='upper left', ncol=1, prop={'size': 15}) #bbox_to_anchor=(-0.5, 1),
        legend.get_frame().set_alpha(0.2) 

        plt.show()
        
    if out:   

        df_out = pd.DataFrame(index=pd.MultiIndex.from_product([['dTm', 'ddG'], ['n_total', f'ungrouped_{stat}', 'n_proteins', f'n_proteins_{stat}', f'avg_{stat}', f'weighted_{stat}', 'runtime (s)']]),
                               columns=[score_name, score_name_2] if score_name_2 != 'tmp' else [score_name], dtype=object) 
        
        for t in ['dTm', 'ddG']:
            
            reduced = i.reset_index()
            reduced = reduced.loc[reduced['type']==t]
            reduced['obs'] = reduced['obs'].astype(int)
            if runtime and score_name not in ['ddG_dir', 'dTm_dir', 'random_dir']:
                runs_reduced = runs.loc[runs['type']==t]

            for score in [score_name, score_name_2]:
                if score != 'tmp':
                    df_out.at[(t, 'n_total'), score] = ungrouped.at[t, 'obs']
                    df_out.at[(t, f'ungrouped_{stat}'), score] = ungrouped.at[t, score]
                    df_out.at[(t, f'n_proteins'), score] = len(db_gt_preds[['code', score_name, t]].dropna().groupby('code').first())
                    df_out.at[(t, f'n_proteins_{stat}'), score] = int(len(reduced.loc[reduced['obs']>=min_obs]))
                    df_out.at[(t, f'avg_{stat}'), score] = reduced[score].mean()
                    df_out.at[(t, f'weighted_{stat}'), score] = np.average(reduced[score], weights=np.log(reduced['obs']))
                    if runtime and score_name not in ['ddG_dir', 'dTm_dir', 'random_dir']:
                        df_out.at[(t, 'runtime (s)'), score] = runs_reduced[f'runtime_{score}'].sum()
        return df_out

    return i


def compute_stats(
    db_gt_preds, 
    split_col=None, split_val=None, split_col_2=None, split_val_2=None, 
    measurements=['ddG', 'dTm'], stats=(), n_classes=2, quiet=False
    ):
    """
    Computes all per-protein and per-dataset stats, including when splitting
    into more than one feature-based scaffold. Splitting is done by specifying
    split_cols (the feature names) and split_vals (the threshold for splitting
    on the respective features). Specifying only split_col and split_val will
    create two scaffolds. Specifying only split_col with split_val > 
    split_val_2 will create 3 scaffolds, with high, intermediate and low values.
    Specifying different split_col and split_col_2 will create 4 scaffolds
    based on high and low values of 2 features. You can pass in a tuple of stats
    to only calculate a subset of the possible stats. You can use n_classes=3
    to eliminate the near-neutral mutations.
    """

    # make sure to not accidentally modify the input
    db_internal = db_gt_preds.copy(deep=True)

    # eliminate the neutral mutations
    if n_classes == 3:
        db_internal = db_internal.loc[
            ~((db_internal['ddG'] > -1) & (db_internal['ddG'] < 1))
            ]
        db_internal = db_internal.loc[
            ~((db_internal['dTm'] > -2) & (db_internal['dTm'] < 2))
            ]

    # case where there are two split_vals on the same column
    if split_col_2 is None and split_val_2 is not None:
        split_col_2 = split_col
    # case where there is no split (default)
    if (split_col is None) or (split_val is None):
        split_col = 'tmp'
        split_val = 0
        db_internal['tmp'] = -1
    # case where there is only one split (2 scaffolds)
    if (split_col_2 is None) or (split_val_2 is None):
        split_col_2 = 'tmp2'
        split_val_2 = 0
        db_internal['tmp2'] = -1

    # there may be missing features for some entries
    db_internal = db_internal.dropna(subset=[split_col, split_col_2])

    # db_discrete will change the continuous measurements into binary labels
    db_discrete = db_internal.copy(deep=True)
    
    # default case
    # stability threshold is defined exactly at 0 kcal/mol or deg. K
    if n_classes == 2:
        if 'ddG' in measurements:
            db_discrete.loc[db_discrete['ddG'] > 0, 'ddG'] = 1
            db_discrete.loc[db_discrete['ddG'] < 0, 'ddG'] = 0
        if 'dTm' in measurements:
            db_discrete.loc[db_discrete['dTm'] > 0, 'dTm'] = 1
            db_discrete.loc[db_discrete['dTm'] < 0, 'dTm'] = 0

    # stabilizing mutations now need to be >= 1 kcal/mol or deg. K
    elif n_classes == 3:
        if 'ddG' in measurements:
            db_discrete.loc[db_discrete['ddG'] > 1, 'ddG'] = 1
            db_discrete.loc[db_discrete['ddG'] < -1, 'ddG'] = -1
        if 'dTm' in measurements:
            db_discrete.loc[db_discrete['dTm'] >= 2, 'dTm'] = 1
            db_discrete.loc[db_discrete['dTm'] <= -2, 'dTm'] = -1

    # for creating a multi-index later
    cols = db_discrete.columns.drop(measurements + [split_col, split_col_2])
    
    # db_discrete_bin has discrete labels and binarized (discrete) predictions
    # drop the split_cols so they do not get binarized
    db_discrete_bin = db_discrete.copy(deep=True).drop(
        [split_col, split_col_2], axis=1).astype(float)

    # binarize predictions (>0 stabilizing, assigned positive prediction)
    db_discrete_bin[db_discrete_bin > 0] = 1
    db_discrete_bin[db_discrete_bin < 0] = 0

    # retrieve the original split_col(s)
    db_discrete_new = db_discrete[
        [split_col] + ([split_col_2] if split_col_2 != split_col else [])]
    # make sure the indices align
    assert all(db_discrete_new.index == db_discrete_bin.index)
    # reunite with split_cols
    db_discrete_bin = pd.concat([db_discrete_bin, db_discrete_new], axis=1)

    # create labels to assign to different scaffolds
    # case no split
    if split_col == 'tmp' and split_col_2 == 'tmp2':
        split = ['']
    # case only one split col
    elif split_col_2 == 'tmp2':
        split = [f'{split_col} > {split_val}', f'{split_col} <= {split_val}']
    # case 2 splits on same col
    elif split_col == split_col_2:
        split = [f'{split_col} > {split_val}', f'{split_val} >= {split_col} > {split_val_2}', f'{split_col} <= {split_val_2}']
    # case 2 splits on 2 cols
    else:
        split = [f'{split_col} > {split_val} & {split_col_2} > {split_val_2}', 
                 f'{split_col} <= {split_val} & {split_col_2} > {split_val_2}',
                 f'{split_col} > {split_val} & {split_col_2} <= {split_val_2}',
                 f'{split_col} <= {split_val} & {split_col_2} <= {split_val_2}']
        
    # separate statistics by measurement, feature scaffold, prediction
    idx = pd.MultiIndex.from_product([['dTm', 'ddG'], split, cols])
    df_out = pd.DataFrame(index=idx)

    # iterate through measurements and splits
    for meas in measurements:
        for sp in split:

            # get new copies that get reduced per scaffold / measurement
            cur_df_bin = db_discrete_bin.copy(deep=True)
            cur_df_discrete = db_discrete.copy(deep=True)
            cur_df_cont = db_internal.copy(deep=True)

            # the following section contains the logic for splitting based on
            # which scaffold is being considered and is self-explanatory
            # there is no logic needed if there is no split requested

            # case where there are 4 scaffolds
            if split_col != 'tmp' and split_col_2 != 'tmp2' and split_col != split_col_2:

                if '>' in sp.split('&')[0]:
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col] > split_val]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col] > split_val]
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col] > split_val]
                elif '<=' in sp.split('&')[0]:
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col] <= split_val]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col] <= split_val]
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col] <= split_val]

                if '>' in sp.split('&')[1]:
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2] > split_val_2]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2] > split_val_2]
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2] > split_val_2]
                elif '<=' in sp.split('&')[1]:
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col_2] <= split_val_2]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col_2] <= split_val_2]
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col_2] <= split_val_2]

            # case where there are 3 scaffolds (on the same feature)
            elif split_col == split_col_2:

                if ('>' in sp and not '>=' in sp):
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col] > split_val]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col] > split_val]
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col] > split_val]
                elif '<=' in sp:
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col] <= split_val_2]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col] <= split_val_2]
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col] <= split_val_2]
                else:
                    cur_df_bin = cur_df_bin.loc[(cur_df_bin[split_col] > split_val_2) & (cur_df_bin[split_col] <= split_val)]
                    cur_df_discrete = cur_df_discrete.loc[(cur_df_discrete[split_col] > split_val_2) & (cur_df_discrete[split_col] <= split_val)]
                    cur_df_cont = cur_df_cont.loc[(cur_df_cont[split_col] > split_val_2) & (cur_df_cont[split_col] <= split_val)]
                    
            # case where there are two scaffolds on one feature
            elif split_col_2 == 'tmp2' and split_col != 'tmp':

                if '>' in sp:
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col] > split_val]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col] > split_val]
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col] > split_val]
                else:
                    cur_df_bin = cur_df_bin.loc[cur_df_bin[split_col] <= split_val]
                    cur_df_discrete = cur_df_discrete.loc[cur_df_discrete[split_col] <= split_val]                  
                    cur_df_cont = cur_df_cont.loc[cur_df_cont[split_col] <= split_val] 
            
            # in this next section we compute the statistics one model at a time
            # all predictions should have the suffix _dir to designate direction mutations
            for col in (tqdm([col for col in cols if ('_dir' in col and not 'runtime' in col)]) \
                if not quiet else [col for col in cols if ('_dir' in col and not 'runtime' in col)]):
                
                # get a reduced version of cur_df_cont for the relevant model
                try:
                    pred_df_cont = cur_df_cont[[col,meas,f'runtime_{col}']].dropna()
                    # we only care about the total runtime for this function
                    df_out.at[(meas,sp,col), 'runtime'] = pred_df_cont[f'runtime_{col}'].sum()
                    pred_df_cont = pred_df_cont.drop(f'runtime_{col}', axis=1)
                except KeyError:
                    pred_df_cont = cur_df_cont[[col,meas]].dropna()
                    df_out.at[(meas,sp,col), 'runtime'] = np.nan    

                # get a reduced version of the classification-task predictions and labels
                pred_df_bin = cur_df_bin[[col,meas]].dropna()
                if 'n' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'n'] = len(pred_df_bin)
                if len(pred_df_bin) == 0:
                    raise AssertionError(f'There are no {col} predictions in this scaffold!')
                
                # compute the 'easy' whole-dataset statistics
                try:
                    tn, fp, fn, tp = metrics.confusion_matrix(pred_df_bin[meas], pred_df_bin[col]).ravel()
                except:
                    tn, fp, fn, tp = 1,1,1,1
                # compute each statistic by default (when stats==())
                if 'tp' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'tp'] = tp
                if 'fp' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'fp'] = fp
                if 'tn' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'tn'] = tn 
                if 'fn' in stats or stats == ():  
                    df_out.at[(meas,sp,col), 'fn'] = fn   
                if 'sensitivity' in stats or stats == (): 
                    df_out.at[(meas,sp,col), 'sensitivity'] = tp/(tp+fn)
                if 'specificity' in stats or stats == ():         
                    df_out.at[(meas,sp,col), 'specificity'] = tn/(tn+fp)
                if 'precision' in stats or stats == (): 
                    df_out.at[(meas,sp,col), 'precision'] = tp/(tp+fp)
                if 'pred_positives' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'pred_positives'] = tp+fp
                if 'accuracy' in stats or stats == (): 
                    df_out.at[(meas,sp,col), 'accuracy'] = metrics.accuracy_score(pred_df_bin[meas], pred_df_bin[col])
                if 'f1_score' in stats or stats == (): 
                    df_out.at[(meas,sp,col), 'f1_score'] = metrics.f1_score(pred_df_bin[meas], pred_df_bin[col])
                if 'MCC' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'MCC'] = metrics.matthews_corrcoef(pred_df_bin[meas], pred_df_bin[col])

                # get a reduced version of the model's predictions with discrete ground truth labels
                pred_df_discrete = cur_df_discrete[[col,meas]].dropna()
                # discrete labels allow testing different thresholds of continuous predictions
                # e.g. for area-under-curve methods
                try:
                    pred_df_discrete[meas] = pred_df_discrete[meas].astype(int)
                    auroc = metrics.roc_auc_score(pred_df_discrete[meas], pred_df_discrete[col])
                    auprc = metrics.average_precision_score(pred_df_discrete[meas], pred_df_discrete[col])
                    if 'auroc' in stats or stats == (): 
                        df_out.at[(meas,sp,col), 'auroc'] = auroc
                    if 'auprc' in stats or stats == (): 
                        df_out.at[(meas,sp,col), 'auprc'] = auprc
                # might fail for small scaffolds
                except Exception as e:
                    if not quiet:
                        print('Couldn\'t compute AUC:', e)
                
                stable_ct = len(pred_df_discrete.loc[pred_df_discrete[meas] > 0])
                if 'n_stable' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'n_stable'] = stable_ct
                
                sorted_preds = pred_df_discrete.sort_values(col, ascending=False).index

                # precision of the top-k predicted-most-stable proteins across the whole slice of data
                for stat in [s for s in stats if 'precision@' in s] if stats != () else ['precision@k']:
                    k = stat.split('@')[-1]
                    if k == 'k':
                        # precision @ k is the fraction of the top k predictions that are actually stabilizing, 
                        # where k is the number of stabilizing mutations in the slice.
                        df_out.at[(meas,sp,col), 'precision@k'] = pred_df_discrete.loc[sorted_preds[:stable_ct], meas].sum() / stable_ct
                    else:
                        k = int(k)
                        if k > stable_ct:
                            print('The number of stabilizing mutations is fewer than k')
                        df_out.at[(meas,sp,col), stat] = pred_df_discrete.loc[sorted_preds[:k], meas].sum() / min(k, stable_ct)

                # using the full (continous) predictions and labels now
                pred_df_cont = cur_df_cont[[col,meas]].dropna()
                pred_df_cont['code'] = pred_df_cont.index.str[:4] 

                # average experimental stabilization of predicted positives
                if 'mean_stabilization' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'mean_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].mean()
                # average experimental stabilization of predicted positives
                if 'net_stabilization' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'net_stabilization'] = pred_df_cont.loc[pred_df_cont[col]>0, meas].sum()
                # average predicted score for experimentally stabilizing mutants
                if 'mean_stable_pred' in stats or stats == ():
                    df_out.at[(meas,sp,col), 'mean_stable_pred'] = pred_df_cont.loc[pred_df_cont[meas]>0, col].mean()

                # top-1 score, e.g. the experimental stabilization achieved on 
                # average for the top-scoring mutant of each protein
                if ('mean_t1s' in stats) or (stats == ()): 
                    top_1_stab = 0
                    for code, group in pred_df_cont.groupby('code'):
                        top_1_stab += group.sort_values(col, ascending=False)[meas].head(1).item()
                    df_out.at[(meas,sp,col), 'mean_t1s'] = top_1_stab / len(pred_df_cont['code'].unique())

                # normalized discounted cumulative gain, a measure of information retrieval ability
                if ('ndcg' in stats) or (stats == ()):
                    # whole-dataset version (not presented in study)
                    df_out.at[(meas,sp,col), 'ndcg'] = compute_ndcg(pred_df_cont, col, meas)
                    cum_ndcg = 0
                    w_cum_ndcg = 0
                    cum_d = 0
                    w_cum_d = 0
                    cum_muts = 0
                    # iterate over unique proteins (wild-type structures)
                    for code, group in pred_df_cont.groupby('code'): 
                        # must be more than one to retrieve, and their stabilities should be different
                        if len(group) > 1 and not all(group[meas]==group[meas][0]):
                            cur_ndcg = compute_ndcg(group, col, meas)
                            # running-total (cumulative)
                            cum_ndcg += cur_ndcg
                            cum_d += 1
                            # weighted running-total (by log(num mutants))
                            w_cum_ndcg += cur_ndcg * np.log(len(group))
                            w_cum_d += np.log(len(group))
                            cum_muts += len(group)
                    df_out.at[(meas,sp,col), 'mean_ndcg'] = cum_ndcg / (cum_d if cum_d > 0 else 1)
                    df_out.at[(meas,sp,col), 'weighted_ndcg'] = w_cum_ndcg / (w_cum_d if w_cum_d > 0 else 1)
                    # may be less than the number of proteins in the dataset based on the if statement               
                    df_out.at[(meas,sp,col), 'n_proteins_ndcg'] = cum_d
                    # may be less than the number of mutants based on the if statement
                    df_out.at[(meas,sp,col), 'n_muts_ndcg'] = cum_muts

                # Spearman's rho, rank-order version of Pearson's r
                # follows same logic as above
                if ('spearman' in stats) or (stats == ()):
                    whole_p, _ = spearmanr(pred_df_cont[col], pred_df_cont[meas])
                    df_out.at[(meas,sp,col), 'spearman'] = whole_p
                    cum_p = 0
                    w_cum_p = 0
                    cum_d = 0
                    w_cum_d = 0
                    cum_muts = 0
                    for code, group in pred_df_cont.groupby('code'):
                        if len(group) > 1 and not all(group[meas]==group[meas][0]):
                            spearman, _ = spearmanr(group[col], group[meas])
                            # can happen if all predictions are the same
                            # in which case ranking ability is poor since we 
                            # already checked that the measurements are different
                            if np.isnan(spearman):
                                spearman=0
                            cum_p += spearman
                            cum_d += 1
                            w_cum_p += spearman * np.log(len(group))
                            w_cum_d += np.log(len(group))
                            cum_muts += len(group)
                    df_out.at[(meas,sp,col), 'mean_spearman'] = cum_p / (cum_d if cum_d > 0 else 1)
                    df_out.at[(meas,sp,col), 'weighted_spearman'] = w_cum_p / (w_cum_d if w_cum_d > 0 else 1)
                    df_out.at[(meas,sp,col), 'n_proteins_spearman'] = cum_d
                    df_out.at[(meas,sp,col), 'n_muts_spearman'] = cum_muts

                # refresh the discrete dataframe
                pred_df_discrete = cur_df_discrete[[col,meas]].dropna()
                pred_df_discrete['code'] = pred_df_discrete.index.str[:4] 
                
                # calculate area under the precision recall curve per protein as with the above stats
                if ('auprc' in stats) or (stats == ()):
                    #df_out.at[(meas,sp,col), 'auprc'] = metrics.average_precision_score(pred_df_discrete[meas], pred_df_discrete[col])
                    cum_ps = 0
                    w_cum_ps = 0
                    cum_d = 0
                    w_cum_d = 0
                    cum_muts = 0
                    for _, group in pred_df_discrete.groupby('code'): 
                        if len(group) > 1:
                            #group[meas] = group[meas].astype(int)
                            cur_ps = metrics.average_precision_score(group[meas], group[col])
                            # NaN if there is only one class in this scaffold for this protein
                            if np.isnan(cur_ps):
                                continue
                            cum_ps += cur_ps
                            cum_d += 1
                            w_cum_ps += cur_ps * np.log(len(group))
                            w_cum_d += np.log(len(group))
                            cum_muts += len(group)
                    df_out.at[(meas,sp,col), 'mean_auprc'] = cum_ps / (cum_d if cum_d > 0 else 1)
                    df_out.at[(meas,sp,col), 'weighted_auprc'] = w_cum_ps / (w_cum_d if cum_d > 0 else 1)
                    df_out.at[(meas,sp,col), 'n_proteins_auprc'] = cum_d
                    df_out.at[(meas,sp,col), 'n_muts_auprc'] = cum_muts

                # these are the expensive statistics (calculated at 100 thresholds)
                # it would take too long to compute them per-scaffold
                if split_col == 'tmp':

                    # the %ile-wise precision score indicates what fraction of 
                    # the proteins in the kth percentile of mutants predicted
                    # to be the most stabilizing (or least destabilizing) are
                    # actually experimentally stabilizing
                    if ('auppc' in stats) or (stats == ()):
                        percentiles = [str(int(s))+'%' for s in range(101)]
                    else:
                        percentiles = [s for s in stats if '%' in s]

                    for stat in percentiles:
                        # in this percentile and stabilizing
                        df_out.at[(meas,sp,col), stat] = 0
                        # number of predicted positives in this percentile
                        df_out.at[(meas,sp,col), 'pos_'+stat] = 0
                        # number of stabilizing mutants experimentally
                        df_out.at[(meas,sp,col), 'stab_'+stat] = 0
                    
                    # percentiles are defined per-protein
                    for name, group in pred_df_cont.groupby('code'):
                        # stabilizing mutations are defined at 0 threshold
                        kth_ground_truth = set(group.loc[group[meas] > 0].index)
                        sorted_predictions = group.sort_values(col, ascending=False)

                        for stat in percentiles:
                            k = stat.split('%')[0]
                            # convert percentile to top-k-percentage
                            k = (100-int(k))/100
                            # always a fraction of the number of muts per-protein
                            l = int(len(group) * k)

                            # top predictions at percentile
                            kth_prediction = set(sorted_predictions.head(l).index)
                            # subset which are stabilizing
                            overlapping_values = kth_ground_truth.intersection(kth_prediction)

                            total = len(overlapping_values)
                            total_pos = len(kth_prediction)

                            # correctly identified (true positives)
                            df_out.at[(meas,sp,col), stat] += total
                            # total positive predictions
                            df_out.at[(meas,sp,col), 'pos_'+stat] += total_pos
                            df_out.at[(meas,sp,col), 'stab_'+stat] += len(kth_ground_truth)

                    for stat in percentiles:
                        # TP / (TP+FP) = precision
                        df_out.at[(meas,sp,col), stat] /= df_out.at[(meas,sp,col), 'pos_'+stat]
                    
                    df_out = df_out.drop(['pos_'+stat for stat in percentiles], axis=1)

                    # estimate the AUC as the mean since the max is 1 and min is 0 and dx stays the same
                    if ('auppc' in stats) or (stats == ()):
                        df_out.at[(meas,sp,col), 'auppc'] = df_out.loc[(meas,sp,col), [c for c in df_out.columns if '%' in c and not 'stab_' in c]].mean()


                    # the k-times recovery score indicates recall, or the number
                    # of experimentally stabilizing mutations recovered, as a
                    # function of the number of top predictions screened
                    if ('aukxrc' in stats) or (stats == ()):
                        # use a log-distributed space to weight AUC toward low-N recovery
                        ks = [str(round(s, 2))+'x_recovery' for s in np.logspace(np.log(0.01), np.log(3), 100, base=np.e)]
                    else:
                        ks = [s for s in stats if 'x_recovery' in s]

                    for k_ in ks:
                        k_factor = float(k_.split('x_recovery')[0])
                        df_out.at[(meas,sp,col), str(k_factor)+'x_recovery'] = 0
                        # number of stabilizing mutants to be recovered
                        df_out.at[(meas,sp,col), 'gt_'+str(k_factor)+'x_recovery'] = 0

                    for name, group in pred_df_cont.groupby('code'):
                        kth_ground_truth = set(group.loc[group[meas] > 0].index)
                        sorted_predictions = group.sort_values(col, ascending=False)

                        for k_ in ks:
                            # k is some multiple of the true number of stabilizing mutations
                            k_factor = float(k_.split('x_recovery')[0])
                            k = len(kth_ground_truth)
                            k *= k_factor
                            k = int(k)
                            if k == 0:
                                continue

                            # top predictions
                            kth_prediction = set(sorted_predictions.head(k).index)
                            overlapping_values = kth_ground_truth.intersection(kth_prediction)

                            total = len(overlapping_values)
                            total_gt = len(kth_ground_truth)

                            df_out.at[(meas,sp,col), str(k_factor)+'x_recovery'] += total
                            df_out.at[(meas,sp,col), 'gt_'+str(k_factor)+'x_recovery'] += total_gt
                        
                    for k_ in ks:
                        k_factor = float(k_.split('x_recovery')[0])
                        # TP / (TP+FN) = recall
                        df_out.at[(meas,sp,col), str(k_factor)+'x_recovery'] /= df_out.at[(meas,sp,col), 'gt_'+str(k_factor)+'x_recovery']
                        
                    df_out = df_out.drop(['gt_'+k_ for k_ in ks], axis=1)

                    # although the mean isn't technically the AUC, this enables the desired weighted AUC
                    if ('aukxrc' in stats) or (stats == ()):
                        df_out.at[(meas,sp,col), 'aukxrc'] = df_out.loc[(meas,sp,col), [c for c in df_out.columns if 'x_recovery' in c]].mean()
            
    hybrid = ['mifst']
    evolutionary = ['tranception', 'msa_transformer', 'esm1v', 'msa']
    structural = ['cartesian_ddg', 'esmif', 'mpnn', 'mif', 'monomer_ddg', 'korpm']

    df_out = df_out.reset_index()
    
    # add labels for the input information used by the model
    #df_out['model_type'] = 'structural'
    for k,v in {'hybrid': hybrid, 'evolutionary': evolutionary, 'structural': structural}.items():
        for m in v:
            # there are many variants of the models so just check if their base name matches
            df_out.loc[df_out['level_2'].str.contains(m), 'model_type'] = k
    df_out = df_out.rename({'level_0': 'measurement', 'level_1': 'class', 'level_2': 'model'}, axis=1)

    df_out = df_out.set_index(['measurement', 'model_type', 'model', 'class'])
    # sort by measurement type, and then model type within each measurement type
    # class is the scaffold
    df_out = df_out.sort_index(level=1).sort_index(level=0)

    return df_out.dropna(how='all')


def is_combined_model(column_name):
    pattern = r'\w+ \+ \w+ \* -?[\d\.]+'
    return bool(re.match(pattern, column_name))


def process_index(index_str):
    # Split the index string by ' + '
    components = index_str.split(' + ')

    # Initialize the model and weight columns
    model1 = None
    weight1 = 1
    model2 = None
    weight2 = 0

    # Process the components
    for component in components:
        model_weight = component.split(' * ')

        if len(model_weight) == 1:
            # Only one model with an implicit weight of 1
            model1 = model_weight[0].strip()
            model2 = model1
        elif len(model_weight) == 2:
            model, weight = model_weight

            if model1 is None:
                model1 = model.strip()
                weight1 = float(weight.strip())
            else:
                model2 = model.strip()
                weight2 = float(weight.strip())

    return model1, weight1, model2, weight2


def calculate_p_values(df, ground_truth_col, statistic, n_bootstraps=100):
    compute = {'auprc': compute_auprc, 'weighted_ndcg': compute_weighted_ndcg, 'weighted_spearman': compute_weighted_spearman, 
        'mean_t1s': compute_t1s, 'mean_stabilization': compute_mean_stab, 'net_stabilization': compute_net_stab}[statistic]

    df_out = pd.DataFrame()
    model_columns = [col for col in df.columns if col != ground_truth_col]
    n_models = len(model_columns)
     # Initialize DataFrame of p-values with NA
    p_values = pd.DataFrame(index=[col for col in model_columns if not is_combined_model(col)], 
                            columns=[col for col in model_columns if not is_combined_model(col)]) 
    p_values = p_values.fillna(np.inf)
    mean_values = pd.DataFrame(index=[col for col in model_columns if not is_combined_model(col)], 
                        columns=[col for col in model_columns if not is_combined_model(col)])
    mean_values = mean_values.fillna(-np.inf)

    # Bootstrap
    bootstrap_statistics = np.zeros((n_bootstraps, n_models))
    for i in tqdm(range(n_bootstraps)):
        df_resampled = df.sample(n=len(df), replace=True)
        for j, model_col in enumerate(model_columns):
            stat = compute(df_resampled, model_col, ground_truth_col)
            bootstrap_statistics[i, j] = stat

    mean_bootstrap_statistics = pd.DataFrame(np.mean(bootstrap_statistics, axis=0), index=model_columns)
    std_bootstrap_statistics = pd.DataFrame(np.std(bootstrap_statistics, axis=0), index=model_columns)

    # Compute p-values for each pair of models
    for j in range(n_models):
        #print(model_columns[j])
        better_combination = False
        model_j_statistic = bootstrap_statistics[:, j]
        if not is_combined_model(model_columns[j]):
            mean_values.loc[model_columns[j], model_columns[j]] = np.mean(model_j_statistic)
            continue
        
        constituent_models = model_columns[j].split(" * ")[0].split(" + ")

        # if the first constituent model has a higher mean value than the second
        if np.mean(bootstrap_statistics[:, model_columns.index(constituent_models[0])]) > np.mean(bootstrap_statistics[:, model_columns.index(constituent_models[1])]):
            model_k_statistic = bootstrap_statistics[:, model_columns.index(constituent_models[0])]
            best_constituent = constituent_models[0]
        # if the second constituent model has a higher mean value than the first
        else:
            model_k_statistic = bootstrap_statistics[:, model_columns.index(constituent_models[1])]
            best_constituent = constituent_models[1]
        t_stat, p_value = stats.ttest_rel(model_j_statistic, model_k_statistic)
        assert all([c in p_values.columns for c in constituent_models])
        df_out = pd.concat([df_out, pd.DataFrame({0:  [constituent_models[0], 1, constituent_models[1], model_columns[j].split(" * ")[1], np.mean(model_j_statistic), p_value]}).T])

        # replace the mean bootstrap value if it is higher than the current value
        if np.mean(model_j_statistic) > mean_values.loc[constituent_models[0], constituent_models[1]]:
            better_combination = True
            mean_values.loc[constituent_models[0], constituent_models[1]] = np.mean(model_j_statistic)
            assert np.mean(model_j_statistic) > mean_values.loc[constituent_models[1], constituent_models[0]]
            mean_values.loc[constituent_models[1], constituent_models[0]] = np.mean(model_j_statistic)

        #print(model_columns[j], np.mean(model_j_statistic), best_constituent, np.mean(model_k_statistic), np.log10(p_value)) 
        # if the mean value for combined model j is greater than its best constituent, and this combination is better than the previous best, record the p_value
        if (np.mean(model_j_statistic) > np.mean(model_k_statistic)) and better_combination:
            #print('better combination')
            # record the p_value in the upper left triangle
            if p_value < p_values.loc[constituent_models[0], constituent_models[1]]:
                #print(p_value, p_values.loc[constituent_models[0], constituent_models[1]]) 
                p_values.loc[constituent_models[0], constituent_models[1]] = p_value
                assert p_value < p_values.loc[constituent_models[1], constituent_models[0]]
                p_values.loc[constituent_models[1], constituent_models[0]] = p_value
            #else:
                #print('no better p-value')

    df_out.columns = ['model1', 'weight1', 'model2', 'weight2', f'mean_{statistic}', 'p_value']
    return p_values, mean_values, df_out

def model_combinations_heatmap(df, statistic, measurement, n_bootstraps=100, threshold=None, custom_colors=dict()):

    font = {'size'   : 14}
    matplotlib.rc('font', **font)

    df_slice = df.xs(measurement, level=0).reset_index()
    #df_slice = df_slice.loc[~df_slice['model'].isin(['ddG_dir', 'dTm_dir'])]
    df_slice[['model1', 'weight1', 'model2', 'weight2']] = df_slice['model'].apply(process_index).apply(pd.Series)
    #df_slice = df_slice.loc[df_slice['weight2'] > 0]
    df = df_slice.loc[df_slice['model1'].isin(custom_colors.keys()) & df_slice['model2'].isin(custom_colors.keys())]
    df = df[['model1', 'weight1', 'model2', 'weight2', statistic, 'runtime']]

    fig, ax = plt.subplots(figsize=(20,15))

    pdf = df.copy(deep=True).reset_index(drop=True).sort_values(statistic, ascending=False).reset_index(drop=True)
    pdf['orig_col'] = pdf['model1'] + ' + ' + pdf['model2'] + ' * ' + pdf['weight2'].astype(str)
    pdf.loc[pdf['weight2']==1, 'orig_col'] = pdf['model1'] + ' + ' + pdf['model2'] + ' * ' + '1'
    pdf.loc[pdf['weight2']==0, 'orig_col'] = pdf['model1']
    pdf = dfm[list(pdf['orig_col'].values)]
    pdf = pdf.join(db_measurements[measurement]).dropna(how='any')

    pvals, performances, stat_df = calculate_p_values(pdf, ground_truth_col=measurement, statistic=statistic, n_bootstraps=n_bootstraps)
    log_pvals = pvals.astype(float).applymap(np.log10)
    diagonal_indices = np.argsort(-performances.values.diagonal())
    performances = performances.iloc[diagonal_indices, diagonal_indices]
    log_pvals = log_pvals.iloc[diagonal_indices, diagonal_indices]
    #print(performances)
    
    runtimes = performances.copy(deep=True)
    for model, row in runtimes.iterrows():
        for model2, val in row.iteritems():
            #print(model1, model2)
            try:
                runtimes.at[model, model2] = df.loc[(df['model1']==model) & (df['model2']==model2), 'runtime'].head(1).item()
            except:
                runtimes.at[model, model2] = 0

    factor = 500
    runtimes = runtimes.where(np.triu(np.ones(performances.shape)).astype(np.bool)).applymap(np.log10) * factor #.applymap(lambda x: '{:.2e}'.format(x)).to_numpy()

    white_cmap = ListedColormap(['white'])
    
    # absolute / mean performance
    sns.heatmap(performances.T.iloc[:, ::-1], annot=True, cmap='viridis', cbar=False, fmt='.2e' if threshold is not None else '.3f', vmin=threshold,
        mask=np.tril(np.ones(log_pvals.shape, dtype=bool), k=-1)[::-1, :], ax=ax, annot_kws={"size": 10} if threshold is not None else None)
    cbar2 = plt.colorbar(ax.collections[0], ax=ax, location="right", use_gridspec=False, pad=-0.05)

    flattened_values = log_pvals[(log_pvals > -10000) & (log_pvals < 10000)].values.ravel()
    flattened_list = [value for value in flattened_values if not np.isnan(value)]

    vmin = min(flattened_list)
    vmax = max(flattened_list)

    # delta / P-value for difference
    sns.heatmap(log_pvals.T.iloc[:, ::-1], annot=True, cmap='coolwarm', cbar=False, fmt='.2f', vmin=vmin, vmax=vmax,
        mask=np.triu(np.ones(log_pvals.shape, dtype=bool))[::-1, :], ax=ax)
    cbar1 = plt.colorbar(ax.collections[1], ax=ax, location="right", use_gridspec=False, pad=0.05)

    # scatter for runtimes
    x, y = np.meshgrid(np.arange(runtimes.T.iloc[:, ::-1].to_numpy().shape[1]), np.arange(runtimes.T.iloc[:, ::-1].to_numpy().shape[0]))
    cmap = plt.get_cmap('viridis')
    plt.scatter(x=[i + 0.5 for i in x.ravel()], y=[i + 0.5 for i in y.ravel()], c='w',
        s=runtimes.T.iloc[:, ::-1].to_numpy().ravel(), edgecolors='k', linewidths=0.5, alpha=0.3)

    #sizes = np.stack([times.to_numpy(), delta_time_df])
    sizes = runtimes.to_numpy()
    # Create custom legend elements for sizes
    #size_min = np.nanmin(sizes[sizes > 0])
    size_min = 1 * factor
    size_max = max(sizes.ravel())
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'\n    10^{(size_min/factor):.1f} seconds \n', markersize=np.sqrt(size_min), markeredgecolor='k'),
                    Line2D([0], [0], marker='o', color='w', label=f'\n    10^{(size_max/factor):.1f} seconds\n', markersize=np.sqrt(size_max), markeredgecolor='k')]

    # Add size legend below the chart
    legend = ax.legend(handles=legend_elements, bbox_to_anchor=(1.3, -0.03), ncol=1, fontsize=14, borderaxespad=1, 
        title='Total Runtime Required (s)')
    legend.get_frame().set_linewidth(0.0)
    legend.get_title().set_position((0, 30)) 
    legend.get_title().set_fontsize(16)

    for tick_label in ax.get_xticklabels():
        for key, val in custom_colors.items():
            if key in tick_label.get_text():
                tick_label.set_color(custom_colors[key])
    for tick_label in ax.get_yticklabels():
        for key, val in custom_colors.items():
            if key in tick_label.get_text():
                tick_label.set_color(custom_colors[key])

    remapped_x = [remap_names_2[tick.get_text()] if tick.get_text() in remap_names_2.keys() else tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticklabels(remapped_x)
    ax.set_yticklabels(remapped_x[::-1])

    measurement_ = {'ddG': 'ΔΔG', 'dTm': 'ΔTm'}[measurement]
    try:
        statistic_ = {'weighted_ndcg': 'wNDCG', 'weighted_spearman': 'wρ', 'weighted_auprc': 'wAUPRC', 'auprc': 'AUPRC', 'mean_t1s': 'mean top-1 score', 'mean_stabilization': 'Mean Stabilization', 'net_stabilization': 'Net Stabilization'}[statistic]
    except:
        statistic_ = statistic
    plt.title(f'Log(P-value) for Change in {statistic_} and Relative Runtimes of Model Combinations for {measurement_}')
    plt.text(1.02, 0.85, f'Max {statistic_} and Total Runtimes of Model Combinations for {measurement_}', va='center', ha='left', fontsize=16, rotation=270, rotation_mode='anchor', transform=plt.gca().transAxes)
    #plt.ylabel('Reference model')
    #plt.xlabel('Added Model')
    plt.ylabel(None)
    plt.xlabel(None)
    plt.show()

    return stat_df, log_pvals

def custom_barplot(data, x, y, hue, width, ax, use_color=None, legend_labels=None, legend_colors=None, std=True):

    if legend_labels is not None and legend_colors is not None:
        lut = dict(zip(legend_labels, legend_colors))

    unique_x = list(data[x].unique())
    data = data.copy(deep=True)#.sort_values(width)
    if legend_labels is not None:
        unique_hue = legend_labels
        unique_width = data.groupby([hue, width]).mean().astype(int).reset_index(level=1).loc[unique_hue][width]
    else:
        unique_hue = data[hue].unique()
        unique_width = data[width].unique()

    #assert len(unique_hue) == len(unique_width)

    if use_color == None:
        colors = legend_colors
    else:
        colors = [use_color]

    max_width = sum(unique_width)

    bar_centers = np.zeros((len(unique_x), len(unique_hue)))
    for i in range(len(unique_x)):
        bar_centers[i, :] = i

    #print(unique_width)

    w_sum = 0
    for j, w in enumerate(unique_width):
        w_sum += w
        bar_centers[:, j] += (-max_width / 2 + w_sum -w/2) / (max_width * 1.1)

    for j, (width_value, hue_value, color) in enumerate(zip(unique_width, unique_hue, colors)):
        y_max = -1
        for i, x_value in enumerate(unique_x):
            if 'ddG' in x_value or 'dTm' in x_value:
                continue
            filtered_data = data[(data[x] == x_value) & (data[width] == width_value)]
            y_std = 0
            if std:
                y_value = filtered_data[f'{y}_mean'].item()#.mean()
                y_std = filtered_data[f'{y}_std'].item()

            y_max = max(y_max, y_value)
            bar_width = filtered_data[width].mean() / (max_width * 1.1)

            if legend_labels is not None and legend_colors is not None:
                color = lut[hue_value]
            ax.bar(bar_centers[i, j], y_value, color=color, width=bar_width, alpha=1 if not use_color else 0.4, yerr=y_std)
        ax.axhline(y=y_max, color=color, linestyle='dashed')

    ax.set_xticks(np.arange(len(unique_x)))
    ax.set_xticklabels(unique_x)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if legend_labels is not None and legend_colors is not None:
        legend_elements = [Patch(facecolor=lut[hue_value], label=f'{hue_value}: {int(width_value)}') for hue_value, width_value in zip(unique_hue, unique_width)]
    else:
        legend_elements = [Patch(facecolor=color, label=f'{hue_value}: {int(width_value)}') for hue_value, width_value, color in zip(unique_hue, unique_width, colors)]
    #ax.legend(handles=legend_elements, title=hue)
    return legend_elements

def compare_performance(dbc,
                        threshold_1 = 1.5, 
                        threshold_2 = None, 
                        split_col = 'hbonds', 
                        split_col_2 = None, 
                        measurement = 'ddG',
                        statistic = 'MCC',
                        statistic_2 = None,
                        n_bootstraps = 100,
                        count_proteins = False,
                        count_muts = False,
                        custom_colors = None
                        ):

    if statistic_2 is None:
        statistic_2 = statistic

    font = {'size'   : 20}
    matplotlib.rc('font', **font)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,20), sharex=True)
    fig.patch.set_facecolor('white')
    sns.set_palette('tab10')

    db_complete = dbc.copy(deep=True)[list(custom_colors.keys()) + [c for c in dbc.columns if '_dir' not in c]]
    db_complete = db_complete.dropna(subset=measurement)

    # Ungrouped performance (doesn't change)
    #ungrouped = compute_stats(db_complete, measurements=[measurement], stats=[statistic_2] + ['n', 'tp', 'tn', 'fp', 'fn'])
    for i in tqdm(range(n_bootstraps)):
        if i == 0:
            full = compute_stats(db_complete.sample(frac=1, replace=True), measurements=[measurement], stats=[statistic] + ['n'])
        else:
            ungrouped = compute_stats(db_complete.sample(frac=1, replace=True), measurements=[measurement], stats=[statistic] + ['n'], quiet=True)
            cur = ungrouped.reset_index()[['n', 'model', 'class', statistic_2] \
                + ([f'n_proteins_{statistic}'] if count_proteins else []) \
                + ([f'n_muts_{statistic}'] if count_muts else [])]
            full = full.merge(cur, on=['model', 'class'], suffixes=('', f'_{i}'))
            #print(full)

    full = full.rename({statistic_2: statistic_2+'_0', 'n': 'n_0'}, axis=1)
    full[f'{statistic_2}_mean'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].mean(axis=1)
    if count_proteins:
        full = full.rename({f'n_proteins_{statistic}': f'n_proteins_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_proteins_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    elif count_muts:
        full = full.rename({f'n_muts_{statistic}': f'n_muts_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_muts_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    else:
        full['n'] = full[[f'n_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    full[f'{statistic_2}_std'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].std(axis=1)
    
    if statistic_2 == 'ndcg':
        full[f'{statistic_2}_mean'] = 100**full[f'{statistic_2}_mean']
        full[f'{statistic_2}_std'] = 100**full[f'{statistic_2}_std']

    #full = full.drop(statistic_2, axis=1)
    ungrouped = full#.rename({f'{statistic_2}_mean': statistic_2}, axis=1)
    #print(ungrouped)
    ungrouped0 = ungrouped
    ungrouped0 = ungrouped0.sort_values(f'{statistic_2}_mean', ascending=False)
    #print(ungrouped0)

    # Unnormalized split performance
    #c = compute_stats(db_complete, split_col=split_col, split_col_2=split_col_2, split_val=threshold_1, split_val_2=threshold_2, measurements=[measurement], stats=[statistic_2] + ['n', 'tp', 'tn', 'fp', 'fn'])
    
    full = pd.DataFrame()
    for i in tqdm(range(n_bootstraps)):
        if i == 0:
            full = compute_stats(db_complete.sample(frac=1, replace=True), 
                split_col=split_col, split_col_2=split_col_2, split_val=threshold_1, split_val_2=threshold_2, measurements=[measurement], stats=[statistic] + ['n'])
        else:
            c = compute_stats(db_complete.sample(frac=1, replace=True), quiet=True,
                split_col=split_col, split_col_2=split_col_2, split_val=threshold_1, split_val_2=threshold_2, measurements=[measurement], stats=[statistic] + ['n'])
            cur = c.reset_index()[['n', 'model', 'class', statistic_2]
                            + ([f'n_proteins_{statistic}'] if count_proteins else []) \
                            + ([f'n_muts_{statistic}'] if count_muts else [])]
            full = full.merge(cur, on=['model', 'class'], suffixes=('', f'_{i}'))
            
    full = full.rename({statistic_2: statistic_2+'_0', 'n': 'n_0'}, axis=1)
    full[f'{statistic_2}_mean'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].mean(axis=1)
    if count_proteins:
        full = full.rename({f'n_proteins_{statistic}': f'n_proteins_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_proteins_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    elif count_muts:
        full = full.rename({f'n_muts_{statistic}': f'n_muts_{statistic}_0'}, axis=1)
        full['n'] = full[[f'n_muts_{statistic}_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    else:
        full['n'] = full[[f'n_{i}' for i in range(n_bootstraps)]].mean(axis=1).astype(int)
    full[f'{statistic_2}_std'] = full[[f'{statistic_2}_{i}' for i in range(n_bootstraps)]].std(axis=1)

    if statistic_2 == 'ndcg':
        full[f'{statistic_2}_mean'] = 100**full[f'{statistic_2}_mean']
        full[f'{statistic_2}_std'] = 100**full[f'{statistic_2}_std']

    splits = full.set_index('model') 
    splits = splits.loc[ungrouped0['model']]#.reset_index()
    splits = splits.loc[ungrouped0['model']].reset_index() #.loc[splits['measurement']==measurement]

    #ungrouped0 = splits.melt(id_vars=['model', 'class'], value_vars=['tp', 'tn', 'fp', 'fn'])

    dbc = db_complete.copy(deep=True)
    
    if split_col_2 is not None:
        dbc[f'{split_col} > {threshold_1} & {split_col_2} > {threshold_2}'] = (dbc[split_col] > threshold_1) & (dbc[split_col_2] > threshold_2)
        dbc[f'{split_col} <= {threshold_1} & {split_col_2} > {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col_2] > threshold_2)
        dbc[f'{split_col} > {threshold_1} & {split_col_2} <= {threshold_2}'] = (dbc[split_col] > threshold_1) & (dbc[split_col_2] <= threshold_2)
        dbc[f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col_2] <= threshold_2)
        vvs = [f'{split_col} > {threshold_1} & {split_col_2} > {threshold_2}', 
                 f'{split_col} <= {threshold_1} & {split_col_2} > {threshold_2}',
                 f'{split_col} > {threshold_1} & {split_col_2} <= {threshold_2}',
                 f'{split_col} <= {threshold_1} & {split_col_2} <= {threshold_2}']
    elif threshold_2 is None:
        dbc[f'{split_col} > {threshold_1}'] = dbc[split_col] > threshold_1
        dbc[f'{split_col} <= {threshold_1}'] = dbc[split_col] <= threshold_1
        vvs = [f'{split_col} > {threshold_1}', f'{split_col} <= {threshold_1}']
    else:
        dbc[f'{split_col} > {threshold_1}'] = dbc[split_col] > threshold_1
        dbc[f'{split_col} > {threshold_1}'] = dbc[split_col] > threshold_1
        dbc[f'{threshold_1} >= {split_col} > {threshold_2}'] = (dbc[split_col] <= threshold_1) & (dbc[split_col] > threshold_2)
        dbc[f'{split_col} <= {threshold_2}'] = dbc[split_col] <= threshold_2
        vvs = [f'{split_col} > {threshold_1}', f'{threshold_1} >= {split_col} > {threshold_2}', f'{split_col} <= {threshold_2}']

    dbc = dbc.dropna(subset=measurement)
    dbc = dbc.melt(id_vars=dbc.columns.drop(vvs), value_vars=vvs)
    dbc = dbc.loc[dbc['value']].rename({'variable':'split'}, axis=1)
    vvs2 = ungrouped0['model'].unique()

    dbc = dbc.melt(id_vars=['split'], value_vars=vvs2)
    std = dbc.groupby('variable')['value'].transform('std')
    dbc['value'] /= std
    ungrouped1 = pd.DataFrame()
    for key in ungrouped0['model'].unique():
        subset = dbc.loc[dbc['variable']==key]
        ungrouped1 = pd.concat([ungrouped1, subset])

    categories = ungrouped0['model'].unique()

    # Class-wise predicted distribution
    my_palette = ["#34aeeb", "#eb9334", "#3452eb", "#eb4634"]
    legend = sns.boxplot(data=ungrouped1,x='variable',y='value',hue=f'split',ax=ax2, palette=my_palette).legend_
                           # split=True if threshold_2 is None else False, bw=0.2, cut=0,
    labels = [t.get_text() for t in legend.texts]
    colors = [c.get_facecolor() for c in legend.legendHandles]

    ax2.set_title('Class-wise predicted distribution')
    ax2.set_ylabel('Stability / log-likelihood')
    ax2.set_xlabel('')
    ax2.grid()
    ax2.axhline(y=0, color='r', linestyle='dashed')

    legend_elements = custom_barplot(data=splits.drop_duplicates(), x='model', y=statistic_2, hue='class', width='n', ax=ax1, legend_colors=colors, legend_labels=labels)
    _ = custom_barplot(data=ungrouped.reset_index().set_index('model').loc[ungrouped0['model']].drop_duplicates().reset_index(), 
                   x='model', y=statistic_2, hue='class', width='n', ax=ax1, use_color='grey')

    statistic_ = {'weighted_ndcg': 'wNDCG', 'weighted_spearman': 'wρ', 'weighted_auprc': 'wAUPRC', 'auprc': 'AUPRC', 'spearman': 'ρ'}[statistic_2]
    ax1.set_title('Class-wise mean performance') #'Delta vs. split mean'
    ax1.set_ylabel(statistic_)
    ax1.grid()
    ax1.set_xlabel('')

    rename_dict = {'delta_kdh': 'Δ hydrophobicity', 'delta_vol': 'Δ volume', 'abs_ASA': 'SASA'}
    new_legend_elements = []
    for legend_element in legend_elements:
        original_label = legend_element.get_label()
        print(original_label)
        for key in rename_dict.keys():
            if key in original_label:
                original_label = original_label.replace(key, rename_dict[key])
        # Update label if it exists in the dictionary
        #if original_label in rename_dict:
                legend_element.set_label(original_label)
        new_legend_elements.append(legend_element)

    ax1.legend(handles=new_legend_elements, loc='lower left')
    ax2.legend(handles=new_legend_elements)
    ax2.set_xticks(ax2.get_xticks(), categories, rotation=45, ha='right')

    for ax in list([ax2, ax1]):
        for tick_label in ax.get_xticklabels():
            for key, val in custom_colors.items():
                if key in tick_label.get_text():
                    tick_label.set_color(custom_colors[key])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    print([tick.get_text() for tick in ax2.get_xticklabels()])
    remapped_x = [remap_names_2[tick.get_text()] if tick.get_text() in remap_names_2.keys() else tick.get_text() for tick in ax2.get_xticklabels()]
    ax.set_xticklabels(remapped_x)

    print(splits.groupby('class')['n'].max())
    plt.show()
    return splits

def stack_frames(dbf):
    db_stack = dbf.copy(deep=True)
    df = db_stack.melt(var_name='pred_col', value_name='value', value_vars=db_stack.columns, ignore_index=False)
    df = df.reset_index()
    df.loc[df['pred_col'].str.contains('dir'), 'direction'] = 'dir'
    df.loc[df['pred_col'].str.contains('inv'), 'direction'] = 'inv'
    assert len(df.loc[df['pred_col'].str.contains('dir') & df['pred_col'].str.contains('inv')]) == 0
    df = df.set_index(['direction', 'uid'])  # Assuming 'index' is the name of uid column
    df['pred_col'] = df['pred_col'].str.replace('_dir', '')
    df['pred_col'] = df['pred_col'].str.replace('_inv', '')
    dbf = df.pivot_table(index=['direction', 'uid'], columns='pred_col', values='value')
    return dbf

def extract_decimal_number(text):
    # Define a regular expression to match decimal numbers
    decimal_regex = r"\d+\.\d+"
    
    # Search for the decimal number in the text using the regular expression
    match = re.search(decimal_regex, text)
    
    # If a match is found, return the float value of the matched string
    if match:
        return float(match.group())
    
    # If no match is found, return NaN
    return np.nan