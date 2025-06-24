"""
Read results from wandb and plot them
"""
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(df, data_name=''):
    """Plot test select accuracy by dataset and model size, with train_split as part of hue."""
    plt.figure(figsize=(12, 6))
    
    # Use both model_size and train_split as part of the hue
    # Use hue to differentiate model_size, train_split, model_type, verifier_size
    df['model_size'] = df['model_size'].astype(str)
    df['train_split'] = df['train_split'].astype(str)
    df['verifier_size'] = df['verifier_size'].astype(str)
    df['hue_label'] = df.astype(str).apply(lambda x: ', '.join(x[['model_size', 'train_split', 'model_type', 'verifier_size']]), axis=1)

    sns.barplot(
        data=df,
        x='model_size',
        y='test_select_accuracy',
        hue='hue_label',
        dodge=True
    )

    plt.xlabel('Dataset')
    plt.ylabel('Test Select Accuracy')
    plt.title('Test Select Accuracy ')
    plt.xticks(rotation=45)
    plt.legend(title='(Model Size, Train Split, Model Type, Verifier Size)', bbox_to_anchor=(1, 1))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"supervised_results_{data_name}.png")
    plt.close()

def plot_results2(df, data_name=''):
    """Plot test select accuracy by dataset and model size, with train_split as part of hue."""
    
    # Use both model_size and train_split as part of the hue
    # Plot the results with train_split in separate rows

    # Create a combined column for verifier_size and model_type
    df['verifier_model'] = df.apply(lambda row: f"model_type {row['model_type']} | verifier size {row['verifier_size']}", axis=1)

    g = sns.catplot(
        data=df,
        x='model_size',
        y='test_select_accuracy',
        col='model_type',
        hue='verifier_model',
        row='train_split',
        kind='bar',
        height=5,
        aspect=1.5
    )

    g.set_axis_labels("Model Size", "Test Select Accuracy")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig(f"supervised_results_{data_name}.png")
    plt.close()


def plot_results3(df, data_name=''):
    """Plot test select accuracy by model size, splitting model_type into different subplots."""

    model_types = sorted(df['model_type'].unique())
    num_subplots = len(model_types)

    fig, axes = plt.subplots(1, num_subplots, figsize=(14, 6), sharey=True)

    if num_subplots == 1:
        axes = [axes]

    for ax, model_type in zip(axes, model_types):
        df_sub = df[df['model_type'] == model_type]

        verifier_sizes = sorted(df_sub['verifier_size'].unique())
        train_splits = sorted(df_sub['train_split'].unique())
        model_sizes = sorted(df_sub['model_size'].unique())

        total_bar_width = 0.8
        bar_width = total_bar_width / (len(verifier_sizes) * len(train_splits))
        x_indices = np.arange(len(model_sizes))

        colors = plt.cm.get_cmap('tab10', len(verifier_sizes))

        for i, verifier_size in enumerate(verifier_sizes):
            for j, split in enumerate(train_splits):
                subset = df_sub[(df_sub['verifier_size'] == verifier_size) & (df_sub['train_split'] == split)]
                positions = x_indices - total_bar_width / 2 + (i * len(train_splits) + j) * bar_width + bar_width / 2

                color = colors(i)
                hatch = '//' if split == 1.0 else ''

                ax.bar(
                    positions,
                    subset['test_select_accuracy'],
                    bar_width,
                    label=f'verifier {verifier_size}, split={split}',
                    hatch=hatch,
                    color=color,
                    edgecolor='black'
                )

        ax.set_xlabel('Model Size')
        ax.set_title(f'Model Type: {model_type}')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(model_sizes, rotation=45)

    axes[0].set_ylabel(f'Test Select Accuracy {data_name}')
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Verifier Size and Train Split')
    plt.suptitle(f'Test Select Accuracy by Model Size ({data_name})')
    plt.tight_layout(rect=[0, 0, 0.85, 0.9], pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.savefig(f"supervised_results_{data_name}.png")
    plt.close()


def hatch_patterns():
    return ['/', '\\', 'x', '-', '+', 'o', 'O', '.', '*']


def hatch_patterns():
    return ['/', '\\', 'x', '-', '+', 'o', 'O', '.', '*']

def hatch_cycle():
    while True:
        for pattern in ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']:
            yield pattern

def read_sweep_results(sweep_name="pssu9o88",model_type_base=''):
    api = wandb.Api()
    sweep = api.from_path(f"ekellbuch/verification/sweeps/{sweep_name}")
    print(sweep.__dict__.keys())


    # Iterate through each run in the sweep
    data = []

    for run in sweep.runs:
        # Retrieve necessary information
        dataset_name = run.config.get('data_cfg.dataset_name')
        model_size = run.config.get('data_cfg.model_size')
        train_split = run.config.get('data_cfg.train_split')
        random_seed = run.config.get('data_cfg.random_seed')
        model_type = run.config.get('model_cfg.model_type', model_type_base)
        # metrics to plot        
        test_select_accuracy = run.summary.get('test_select_accuracy')  # Replace 'accuracy' with the actual metric name if different
        train_select_accuracy = run.summary.get('train_select_accuracy')
        mv_as_verifier = run.config.get('data_cfg.mv_as_verifier')
        fit_type = run.config.get('fit_cfg.fit_type')
        verifier_size = run.config.get('verifier_cfg.verifier_size')
        normalize_type = run.config.get('data_cfg.normalize_type')
        normalize_method = run.config.get('data_cfg.normalize_method')
        model_class = run.config.get('model_cfg.model_class')
        # Append to data list
        data.append({
            'dataset_name': dataset_name,
            'model_size': model_size,
            'train_select_accuracy': test_select_accuracy,
            'test_select_accuracy': test_select_accuracy,
            'train_split': train_split,
            'random_seed': random_seed,
            'model_type': model_type,
            'mv_as_verifier' : mv_as_verifier,
            'fit_type' : fit_type,
            'verifier_size' : verifier_size,
            'normalize_type' : normalize_type,
            'normalize_method' : normalize_method,
            'model_class' : model_class,
        })

    # Convert list to DataFrame
    df = pd.DataFrame(data)
    return df




# Helper function to create lighter colors for 8B verifiers
def lighten_color(color, amount=0.5):
    """
    Returns a lighter version of the specified color.
    
    Parameters:
    color: A matplotlib color string or rgb tuple
    amount: Amount to lighten (0-1), where 1 is white
    """
    import matplotlib.colors as mc
    import colorsys
    
    try:
        c = mc.to_rgb(color)
        c = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    except:
        return color
    

def plot_results4(df, data_name=''):
    """Plot test select accuracy by model size, splitting model_type into different subplots."""

    train_splits = sorted(df['train_split'].unique())
    num_subplots = len(train_splits)

    fig, axes = plt.subplots(1, num_subplots, figsize=(14, 6), sharey=True)

    if num_subplots == 1:
        axes = [axes]

    # hatch pattern by verifier_size
    color_map = {
       "(logistic_regression, 8)": "#f4a582",
       "(majority_vote1, 8)": "#92c5de",
        "(naive_bayes, 8)": "#b8e186",
       "(logistic_regression, 80)": "#d6604d",
       "(majority_vote1, 80)": "#4393c3",
        "(naive_bayes, 80)": "#66bd63",    }    

    # Create light and dark versions for 8B and 80B verifiers
    train_splits = [1.0, 0.8]
    for ax, train_split in zip(axes, train_splits):
        df_sub = df[df['train_split'] == train_split]

        verifier_sizes = sorted(df_sub['verifier_size'].unique())
        #verifier_sizes = [8, 80]
        model_types = sorted(df_sub['model_type'].unique())
        model_sizes = sorted(df_sub['model_size'].unique())
    
        total_bar_width = 0.8
        #bar_width = total_bar_width / (len(verifier_sizes) * len(model_types) * len(model_sizes))
        x_indices = np.arange(len(model_sizes))
        group_width = 0.7  # Width for all bars of a model size
        model_size_gap = 0.1  # Gap between different model sizes
        verifier_group_width = group_width / len(verifier_sizes)  # Width for each verifier group

        # color by model_type
        colors = plt.cm.get_cmap('tab10', len(model_types))

        # for each model size and verifier size
        for i, verifier_size in enumerate(verifier_sizes):
            # offset for each verifier group
            verifier_offset = -group_width/2 + i * verifier_group_width + verifier_group_width/2 

            for j, model_type in enumerate(model_types):
                subset = df_sub[(df_sub['verifier_size'] == verifier_size) & (df_sub['model_type'] == model_type)]
                
                if subset.empty:
                    continue
                # Position bars within verifier group
                bar_width = verifier_group_width / len(model_types)
                positions = i*model_size_gap + x_indices + verifier_offset + (j - len(model_types)/2 + 0.5) * bar_width
            
                # color by model_type
                #color = color_map[model_type]
                
                # hatch patten by verifier_size
                #hatch = hatch_patterns()[i]
                color = color_map[f"({model_type}, {verifier_size})"]

                ax.bar(
                    positions,
                    subset['test_select_accuracy'],
                    bar_width,
                    label=f'verifier size {verifier_size}, model_type={model_type}',
                    #hatch=hatch,
                    color=color,
                    edgecolor='black'
                )

        ax.set_xlabel('Model Size')
        ax.set_title(f'Train Split: {train_split}')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(model_sizes, rotation=45)

    axes[0].set_ylabel(f'Test Select Accuracy {data_name}')
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Verifier Size and Train Split')
    plt.suptitle(f'Test Select Accuracy by Train Split ({data_name})')
    plt.tight_layout(rect=[0, 0, 0.85, 0.9], pad=2.0, h_pad=2.0, w_pad=2.0)
    plt.savefig(f"supervised_results4_{data_name}.png")
    plt.close()




def main():
    
    sweep_name1 = "pssu9o88" 
    sweep_name2 = "l81cqaot" # majority vote where majority_select = "one_sample"
    sweep_name3= "yd10wi7q" # majority vote where majority_select = "majority"
    
    default_names = ["logistic_regression", "majority_vote1", "majority_vote_M"]
    sweep_names = [sweep_name1, sweep_name2, sweep_name3]
    #sweep_names = [sweep_name1, sweep_name2]
    #default_names = ["logistic_regression", "majority_vote1"]
    all_sweep_names = '_'.join(sweep_names)
    outname  = f"supervised_results_sweep_{all_sweep_names}.csv"
    if os.path.exists(outname):
        df3 = pd.read_csv(outname)
    else:
        all_dfs = []
        for i, (sweep_name, default_name) in enumerate(zip(sweep_names, default_names)):
            df = read_sweep_results(sweep_name, model_type_base=default_name)
            #df['model_type'] = default_name
            all_dfs.append(df)
        df3 = pd.concat(all_dfs)

        # Plot the results 
        df3.to_csv(outname, index=False)

    # Filter only for MATH-500
    # FOr each dataset:
    all_datasets = df3['dataset_name'].unique()
    for dataset_name in all_datasets:
        # ignore datasets which are running 
        print(f"Plotting {dataset_name}")
        df = df3[df3['dataset_name'] == dataset_name]

        # DO this filtering only for model_types = 'logistic_regression' and 'naive_bayes',
        # and for other model types, do not filter
        filter_model_types = {'logistic_regression', 'naive_bayes'}
        mask_excluded = df['model_type'].isin(['majority_vote1'])
        # Identify rows that require filtering
        mask_filtered = df['model_type'].isin(filter_model_types)

        # Apply filters only to the required model types
        df = df[mask_excluded | ~mask_filtered | (df['normalize_type'] == 'per_problem')]
        df = df[mask_excluded | ~mask_filtered | (df['normalize_method'] == 'quantile')]
        df = df[mask_excluded | ~mask_filtered | (df['model_class'] == 'per_problem')]
        df = df[mask_excluded | ~mask_filtered | (df['mv_as_verifier'] == True)]

        # Apply `fit_type` filter only for model types that require filtering
        df = df[mask_excluded | ~mask_filtered | (df['fit_type'] == 'search_weights')]
        # Drop NaN values (if any)
        mask_excluded = df['model_type'].isin(['majority_vote_M'])
        df = df[~mask_excluded]
        df['model_size'] = df['model_size'].astype('category')
        df['train_split'] = df['train_split'].astype('category')
        df['mv_as_verifier'] = df['mv_as_verifier'].astype('category')
        df['fit_type'] = df['fit_type'].astype('category')
        df['verifier_size'] = df['verifier_size'].astype('category')
        df['normalize_type'] = df['normalize_type'].astype('category')
        df['normalize_method'] = df['normalize_method'].astype('category')
        df['model_class'] = df['model_class'].astype('category')
        df['model_type'] = df['model_type'].astype('category')

        group_by = ['model_size', 'train_split','verifier_size','model_type']
        df = df.groupby(group_by)['test_select_accuracy'].mean().reset_index()
        dataset_name = dataset_name + "_" + "search_weights"
        plot_results4(df, dataset_name)

                    

if __name__ == "__main__":
    main()