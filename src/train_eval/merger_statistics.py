import torch
import numpy as np

import os
from pathlib import Path
import matplotlib.pyplot as plt


# Returns statistics for each token
def var_mean(tensor: torch.Tensor):
    return torch.mean(torch.var(tensor, dim=0), dim=-1)


# Returns statistics for each token
def mean(tensor: torch.Tensor):
    return torch.mean(tensor, dim=(0,2))



# Extracting features from a merger model
def get_merger_stats(merger, dataset, models_to_merge):
    merger = merger.to('cuda:0')
    merger.eval()

    stats = {}
    with torch.no_grad():
        outputs_merged_full = []
        outputs_full = []
        for i, batch in enumerate(dataset.val_loader):
            inputs = batch['images'].to('cuda:0')
            outputs_merged_full.append(merger(inputs, for_features=True)) # [B, seq_length, new_emb]
            outputs_full.append(merger(inputs))  # [B, seq_length, emb * num_models]

        outputs_merged_full = torch.cat(outputs_merged_full, dim=0)
        outputs_full = torch.cat(outputs_full, dim=0)

        stats['mean_outputs_merged'] = mean(outputs_merged_full).cpu()
        stats['var_outputs_merged'] = var_mean(outputs_merged_full).cpu()

        stats['mean_outputs'] = mean(outputs_full).cpu()
        stats['var_outputs'] = var_mean(outputs_full).cpu()

        outputs_split = torch.chunk(outputs_full, len(models_to_merge), dim=-1)
        for i, output in enumerate(outputs_split):
            stats[f'mean_output_{models_to_merge[i]}'] = mean(output).cpu()
            stats[f'var_output_{models_to_merge[i]}'] = var_mean(output).cpu()

    return stats





def create_stats_plots(before_train_stats, after_train_stats, plots_path, layer_name, layer_num=0):

    if before_train_stats is None or after_train_stats is None:
        return

    plots_path = os.path.join(plots_path, 'stats_plots')
    Path(plots_path).mkdir(parents=True, exist_ok=True)

    for key in before_train_stats.keys():
        plot_tensors(A=before_train_stats[key],
                     B=after_train_stats[key],
                     title="Stats during training - {} ({} {})".format(key, layer_name, layer_num),
                     path=os.path.join(plots_path, '{}_{}'.format(layer_name, layer_num)),
                     plot_name= '{}.png'.format(key))

        plot_changed_elements(A=before_train_stats[key],
                     B=after_train_stats[key],
                     title="Stats during training - {} ({} {})".format(key, layer_name, layer_num),
                     path=os.path.join(plots_path, '{}_{}'.format(layer_name, layer_num)),
                     plot_name='worse_{}.png'.format(key))







####################################
########### PLOTTING ###############


def plot_tensors(A, B, title, path, plot_name):
    """
    Plots two torch tensors A and B as columns in a matrix plot.

    Args:
    A (torch.Tensor): First tensor of length n.
    B (torch.Tensor): Second tensor of length n.
    title (str): Title of the plot.
    path (str): Path where the plot will be saved.
    """
    if A.shape != B.shape:
        raise ValueError("Tensors A and B must have the same shape")

    Path(path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(path, plot_name)

    # Convert tensors to numpy arrays for plotting
    A_np = A.numpy()
    B_np = B.numpy()

    # Create a matrix for plotting
    matrix = np.column_stack((A_np, B_np))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 15))  # Adjust the figsize to make the plot high
    cax = ax.matshow(matrix, aspect='auto')
    cbar = fig.colorbar(cax)

    # Set the axis labels with increased font size
    ax.set_ylabel("Sequence Length", fontsize=14)
    ax.set_xlabel("Training", fontsize=14)

    # Set the x-ticks and move them to the bottom
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After"], fontsize=14)
    ax.xaxis.set_ticks_position('bottom')  # Move x-ticks to the bottom

    # Set the y-ticks with increased font size
    ax.set_yticks(range(0, len(A), 25))
    ax.set_yticklabels(range(0, len(A), 25), fontsize=12)

    # Set the title with increased font size
    plt.title(title, fontsize=18)

    # Increase font size for color bar labels
    cbar.ax.tick_params(labelsize=12)

    # Save the plot
    plt.savefig(path)
    plt.close()



def plot_changed_elements(A, B, title, path, plot_name):
    """
    Plots the elements with the largest and smallest changes between two tensors.

    Args:
    A (torch.Tensor): First tensor of length n.
    B (torch.Tensor): Second tensor of length n.
    title (str): Title of the plot.
    path (str): Path where the plot will be saved.
    """
    if A.shape != B.shape:
        raise ValueError("Tensors A and B must have the same shape")

    path = os.path.join(path, plot_name)

    with torch.no_grad():
        # Calculate the absolute differences
        differences = torch.abs((A - B)/A)

        # Find the indices of the five largest changes
        largest_changes_indices = torch.topk(differences, 6).indices

        # Find the index of the smallest change
        smallest_change_index = torch.argmin(differences)

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    x_labels = ['Before', 'After']

    # Plot the elements with the largest changes
    for idx in largest_changes_indices:
        plt.plot([A[idx], B[idx]], label=f'Index {idx}', marker='o')

    # Plot the element with the smallest change
    plt.plot([A[smallest_change_index], B[smallest_change_index]], label='Smallest Change - {}'.format(smallest_change_index), linestyle='--', marker='o')

    # Adding labels and title
    plt.xticks([0, 1], x_labels)
    plt.xlabel("Training")
    plt.ylabel("Value of Elements")
    plt.title(title, fontsize=14)
    plt.legend()

    # Save the plot
    plt.savefig(path)
    plt.close()