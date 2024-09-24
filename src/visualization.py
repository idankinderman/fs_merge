import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import math
import numpy as np
from typing import List, Dict, Optional, Tuple

# Plotting the loss graphs
def save_train_loss_plots(loss_list, path_to_save, title):
    plt.plot(loss_list, linestyle='solid')
    plt.title('Train Loss - {}'.format(title))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    path_fig = os.path.join(path_to_save, '{}_loss.png'.format(title))
    plt.savefig(path_fig)
    plt.close()

    plt.yscale("log")
    plt.plot(loss_list, linestyle='solid')
    plt.title('Train Loss - {} - log scale'.format(title))
    plt.xlabel('Iteration')
    plt.ylabel('Loss - log scale')
    path_fig = os.path.join(path_to_save, '{}_loss_log.png'.format(title))
    plt.savefig(path_fig)
    plt.close()


# Plotting the accuracy graphs
def save_acc_plot(train_acc, val_acc, test_acc, path_to_save, title):
    colors = ['orange', 'red', 'magenta', 'purple', 'black', 'blue', 'skyblue', 'cyan', 'lime', 'green', 'darkgreen']
    colors = ['purple', 'skyblue', 'lime']

    plt.plot(train_acc, label='train', color=colors[0], linestyle='solid', marker='o')
    plt.plot(val_acc, label='val', color=colors[1], linestyle='solid', marker='o')
    plt.plot(test_acc, label='test', color=colors[2], linestyle='solid', marker='o')

    plt.title('Accuracy on {} during training'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(title='data type')
    path_fig = os.path.join(path_to_save, '{}_acc.png'.format(title))
    plt.savefig(path_fig)
    plt.close()


# The plots for the loss curves of each merge layer, during training.
def merge_training_plots(train_loss: List[float],
                         train_loss_each_epoch: List[float],
                         val_loss_each_epoch: Dict[str,List[float]],
                         end_of_epoch_step_num: List[int],
                         horizontal_line: float,
                         title: str,
                         save_path: str,
                         train_inner_loss: Dict[str,List[float]] = None,
                         val_inner_loss_each_epoch: Dict[str, List[float]] = None,
                         only_full: bool=False,
                         features_scales: Dict[str, float]=None) -> None:
    """
    Plots the training loss, train loss at the end of each epoch, and test loss at the end of each epoch.

    Args:
    - train_loss (List[float]): Training loss values.
    - train_loss_each_epoch (List[float]): Training loss values at the end of each epoch.
    - train_inner_loss (Dict[str : List[float]]): Dictionaries of training inner losses values at the end of each epoch.
    - val_loss_each_epoch Dict[str : List[float]]: Dictionaries of test loss values at the end of each epoch.
    - val_inner_loss_each_epoch Dict[str : List[float]]: Dictionaries of test inner losses values at the end of each epoch.
    - end_of_epoch_step_num (List[int]): Step numbers at which each epoch ends.
    - features_scales (Dict[str, float]): Approximation of the scale of each task features.
    - title (str): Title for the plot.
    - save_path (str): Path to save the plot.
    - only_full (bool): Whether to only plot the full val loss.
    """

    colors = [
        'red', 'darkorange', 'yellowgreen', 'green', 'dodgerblue', 'blue', 'blueviolet',
        'mediumorchid', 'mediumorchid', 'magenta', 'mediumvioletred', 'darkcyan'
    ]


    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    if train_loss and val_loss_each_epoch:
        max_val_loss = max(max(losses) for losses in val_loss_each_epoch.values())
        max_train_loss = max(train_loss)
        max_value = max(max_val_loss, max_train_loss)
    elif train_loss:
        max_value = max(train_loss)
    elif val_loss_each_epoch:
        max_value = max(max(losses) for losses in val_loss_each_epoch.values())
    else:
        max_value = None

    #for idx, plot_type in enumerate(['linear', 'log']):
    for idx, plot_type in enumerate(['linear']):
        #ax = ax1 if plot_type == 'linear' else ax2
        ax = ax1

        if plot_type == 'log':
            ax.set_yscale("log")
            if max_value:
                ax.set_ylim(bottom=1e-6, top=max_value*1.01)
        else:
            if max_value:
                ax.set_ylim(bottom=0, top=max_value*1.01)

        # Plotting the training loss
        if train_loss is not None:
            ax.plot(train_loss, label='Training Loss', color='red')

        if train_inner_loss is not None:
            for i, key in enumerate(train_inner_loss.keys()):
                ax.plot(train_inner_loss[key], label=f'{key}', color=colors[i+2], linestyle='--')

        # Plotting the train loss at the end of each epoch
        if train_loss_each_epoch is not None:
            ax.scatter(end_of_epoch_step_num, train_loss_each_epoch, color='red', label='Train Loss at Epoch End', marker='o')

        # Plotting the val loss at the end of each epoch
        if val_loss_each_epoch is not None:
            for i, key in enumerate(val_loss_each_epoch.keys()):
                if 'full' not in key:
                    continue
                ax.scatter(end_of_epoch_step_num, val_loss_each_epoch[key], label=f'Val {key} at Epoch End',
                           marker='x', color=colors[i])

        if val_inner_loss_each_epoch is not None:
            for i, key in enumerate(val_inner_loss_each_epoch.keys()):

                ax.scatter(end_of_epoch_step_num, val_inner_loss_each_epoch[key], label=f'{key} at Epoch End',
                           marker='^', color=colors[i+5])

        if horizontal_line is not None:
            ax.axhline(y=horizontal_line, color='black', linestyle='--', label='Average Merge Model')

        # Edit the legend box
        handles, labels = ax.get_legend_handles_labels()
        """
        if features_scales is not None:
            for key, value in features_scales.items():
                # Create a proxy artist for the new label
                handle = Line2D([], [], color='none', marker='none', linestyle='none',
                                label="{} scale = {:.3f}".format(key, value))
                handles.append(handle)
        """
        ax.legend(handles=handles, loc='upper right')

        # Setting the title
        full_title = title if plot_type == 'linear' else title + ' (log scale)'
        ax.set_title(full_title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss - MSE')
        #ax.legend()

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(save_path + '_loss.png')
    plt.close()


def mul_training_plots(train_loss, end_of_epoch_step_num, val_loss_each_epoch, title, save_path):
    """
    Plots training and validation loss data from dictionaries.

    Parameters:
        train_loss (dict): Dictionary with keys as labels and values as lists of numbers representing training loss.
        end_of_epoch_step_num (list): List of step numbers corresponding to the end of each epoch.
        val_loss_each_epoch (dict): Dictionary with keys as labels and values as lists of numbers representing validation loss at each epoch.
        title (str): Title for the plot.
        save_path (str): Path to save the plot image file.

    Returns:
        None. Saves the plot to the specified path and displays it.
    """
    # Colors for the plot
    colors = [
        'red', 'darkorange', 'yellowgreen', 'green', 'dodgerblue', 'blue', 'blueviolet',
        'mediumorchid', 'magenta', 'mediumvioletred', 'darkcyan'
    ]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plotting training loss
    for idx, (key, values) in enumerate(train_loss.items()):
        ax.plot(values, label=key + ' Train', color=colors[idx % len(colors)])

    # Plotting validation loss
    for idx, (key, values) in enumerate(val_loss_each_epoch.items()):
        ax.scatter(end_of_epoch_step_num, values, label=key + ' Validation', color=colors[idx % len(colors)])

    # Setting labels, title and axis names
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Loss - MSE', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)

    # Saving the plot
    plt.savefig(save_path)
    plt.show()


def plot_multiple_losses(losses_list: List[List[float]],
                         test_loss_each_epoch: List[List[float]],
                         end_of_epoch_step_num: List[List[int]],
                         labels: List[str],
                         title: str,
                         save_path: str,
                         x_axis_name: str = 'Step',
                         y_axis_name: str = 'Train Loss - MSE') -> None:
    """
    Plot multiple loss curves on a single figure.

    Parameters:
    - losses_list: List of lists containing loss values.
    - test_loss_each_epoch: List of lists containing test loss values at the end of each epoch.
    - end_of_epoch_step_num: List of lists of step numbers at which each epoch ends.
    - labels: List of labels for each loss curve.
    - x_axis_name: Name for the x-axis.
    - y_axis_name: Name for the y-axis.
    - title: Title for the entire plot.
    - save_path: Path to save the figure.
    """

    # List of colors in a gradient from red to violet
    colors = [
        'red', 'darkorange', 'gold', 'yellowgreen', 'green',
        'mediumseagreen', 'deepskyblue', 'dodgerblue', 'blue',
        'mediumslateblue', 'blueviolet', 'mediumorchid',
        'magenta', 'mediumvioletred', 'darkcyan'
    ]


    # Check if the number of labels matches the number of loss lists
    if len(losses_list) != len(labels):
        raise ValueError("Number of loss lists must match the number of labels.")

    # Check if there are more loss lists than colors
    if len(losses_list) > len(colors):
        raise ValueError("Number of loss lists exceeds available colors.")

    # Create a new figure
    plt.figure()

    # Determine the maximum y-value across all loss lists
    if losses_list[0] and test_loss_each_epoch[0]:
        max_y_value = max([max(losses) for losses in losses_list] +  [max(losses) for losses in test_loss_each_epoch])
    elif losses_list[0]:
        max_y_value = max([max(losses) for losses in losses_list])
    elif test_loss_each_epoch[0]:
        max_y_value = max([max(losses) for losses in test_loss_each_epoch])
    else:
        raise ValueError("No loss values provided.")

    # Plot each loss list with its corresponding label and color
    if losses_list[0]:
        for i, losses in enumerate(losses_list):
            plt.plot(losses, label=labels[i], color=colors[i])

    # Plot each test loss list with its corresponding label and color
    if test_loss_each_epoch[0]:
        for i in range(len(test_loss_each_epoch)):
            plt.scatter(end_of_epoch_step_num[i], test_loss_each_epoch[i], color=colors[i], marker='x')

    # Set x-axis and y-axis labels, title, and legend
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.ylim(0, max_y_value)  # Set y-axis limits
    plt.legend()

    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close()


def plot_multiple_losses_with_lines(test_loss_each_epoch: List[List[float]],
                                      end_of_epoch_step_num: List[List[int]],
                                      labels: List[str],
                                      title: str,
                                      save_path: str,
                                      list_lines: List[int] | None = None,
                                      list_lines_dashed: List[int] | None = None,
                                      list_lines_label: str | None = None,
                                      list_lines_dashed_label: str | None = None,
                                      x_axis_name: str = 'Step',
                                      y_axis_name: str = 'Train Loss - MSE') -> None:
    """
    Plot multiple loss curves on a single figure.

    Parameters:
    - test_loss_each_epoch: List of lists containing test loss values at the end of each epoch.
    - end_of_epoch_step_num: List of lists of step numbers at which each epoch ends.
    - list_lines: List of y-values at which to draw a vertical line.
    - list_lines_dashed: List of y-values at which to draw a dashed vertical line.
    - labels: List of labels for each loss curve.
    - list_lines_label: One label for the vertical lines.
    - list_lines_dashed_label: One label for the vertical dashed lines.
    - x_axis_name: Name for the x-axis.
    - y_axis_name: Name for the y-axis.
    - title: Title for the entire plot.
    - save_path: Path to save the figure.
    """

    # List of colors in a gradient from red to violet
    colors = [
        'red', 'darkorange', 'gold', 'yellowgreen', 'green',
        'mediumseagreen', 'deepskyblue', 'dodgerblue', 'blue',
        'mediumslateblue', 'blueviolet', 'mediumorchid',
        'magenta', 'mediumvioletred', 'darkcyan'
    ]

    # Create a new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Determine the maximum y-value across all loss lists
    max_y_value = max([max(losses) for losses in test_loss_each_epoch])
    maxed_y_line = max(list_lines + list_lines_dashed)

    # Plot each test loss list with its corresponding label and color
    for i in range(len(test_loss_each_epoch)):
        ax1.scatter(end_of_epoch_step_num[i], test_loss_each_epoch[i], label=labels[i], color=colors[i], marker='x')

    # Plot vertical lines
    for i, line in enumerate(list_lines):
        ax2.axhline(y=line, color=colors[i])
    for i, line in enumerate(list_lines_dashed):
        ax2.axhline(y=line, color=colors[i], linestyle='dashed')

    # Get the existing legend handles and labels
    handles, labels = ax1.get_legend_handles_labels()

    # Create custom legend entries
    line = Line2D([0], [0], color='black', linestyle='-', label=list_lines_label)
    dashed_line = Line2D([0], [0], color='black', linestyle='--', label=list_lines_dashed_label)

    # Add the custom legend entries to the existing handles and labels
    handles.extend([line, dashed_line])
    labels.extend([list_lines_label, list_lines_dashed_label])

    # Display the updated legend
    # Place the legend box outside the plot, to the right
    ax1.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    # Set x-axis and y-axis labels, title, and legend
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)

    fig.suptitle(title)

    ax1.set_ylim(0, max_y_value*1.05)  # Set y-axis limits for the first subplot
    ax2.set_ylim(0, maxed_y_line*1.05)  # Set y-axis limits for the second subplot

    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close()



###############################
def plot_features_stats2(data_dict, save_path, title):
    """
    Create a line plot from the given dictionary, split into four subplots, and save it to the specified path.

    Parameters:
    - data_dict: Dictionary with string keys and float values.
    - save_path: Path to save the figure.
    - title: Title for the figure.
    """

    # Extract keys and values from the dictionary
    labels = list(data_dict.keys())
    values = list(data_dict.values())

    # Calculate the number of elements in each quarter
    quarter_length = math.ceil(len(labels) / 4)

    # Create the figure and subplots with a larger size
    fig, axs = plt.subplots(4, 1, figsize=(12, 15))
    fig.suptitle(title)

    for i in range(4):
        start_idx = i * quarter_length
        end_idx = (i + 1) * quarter_length
        axs[i].plot(labels[start_idx:end_idx], values[start_idx:end_idx], marker='o', linestyle='-')
        axs[i].set_ylabel('Values')
        if i == 3:
            axs[i].set_xlabel('Keys')
        axs[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close()


def plot_features_stats(dict_dicts, save_path, title, colors=None, linestyles=None, vertical_labels=None,
                        num_subplots=4, log_scale=False):
    """
    Create a line plot from the given dictionary of dictionaries, split into num_subplots subplots, and save it to the specified path.

    Parameters:
    - dict_dicts: Dictionary of dictionaries with string keys and float values.
    - save_path: Path to save the figure.
    - title: Title for the figure.
    - vertical_labels: List of labels where vertical lines should be added.
    """

    if colors is None:
        colors = ['red', 'darkorange', 'green', 'deepskyblue', 'blue', 'blueviolet', 'magenta', 'mediumvioletred']

        colors = [
            'red', 'darkorange', 'gold', 'yellowgreen', 'green',
            'mediumseagreen', 'deepskyblue', 'dodgerblue', 'blue',
            'mediumslateblue', 'blueviolet', 'mediumorchid',
            'magenta', 'mediumvioletred', 'darkcyan'
        ]

    if linestyles is None:
        linestyles = ["solid" for i in range(len(dict_dicts))]

    # Assuming all inner dictionaries have the same keys, extract labels from the first one
    sample_dict = next(iter(dict_dicts.values()))
    labels = list(sample_dict.keys())

    # Calculate the number of elements in each quarter
    quarter_length = math.ceil(len(labels) / num_subplots)

    # Create the figure and subplots with a larger size
    fig, axs = plt.subplots(num_subplots, 1, figsize=(18, int(20*(num_subplots/4)+1)))
    fig.suptitle(title)

    key_num = -1
    for key, data_dict in dict_dicts.items():
        key_num += 1
        values = list(data_dict.values())
        for i in range(num_subplots):
            start_idx = i * quarter_length
            end_idx = (i + 1) * quarter_length
            axs[i].plot(labels[start_idx:end_idx], values[start_idx:end_idx],
                        marker='o', linestyle=linestyles[key_num], label=key, color=colors[key_num])

            if log_scale:
                axs[i].set_yscale('log')

            # Add vertical lines for specified labels
            if vertical_labels is not None:
                for v_label in vertical_labels:
                    if v_label in labels[start_idx:end_idx]:
                        axs[i].axvline(x=v_label, color='black', linestyle='--')

            axs[i].set_ylabel('Values')
            if i == num_subplots-1:
                axs[i].set_xlabel('Keys')
            axs[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
            axs[i].legend()  # Display legend

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close()




#########

def plot_variance_and_entropy(
        train_var: Optional[Dict[str, List[float]]],
        test_var: Dict[str, List[float]],
        train_entropy: Optional[Dict[str, List[float]]],
        test_entropy: Dict[str, List[float]],
        end_of_epoch_step_num: List[int],
        features_scales: Dict[str, float],
        figure_title: str,
        save_path: str
        ) -> None:
    # Create a list of colors that form a gradient
    colors = [
        'red', 'darkorange', 'yellowgreen', 'green', 'dodgerblue', 'blue',
        'mediumslateblue', 'blueviolet', 'mediumorchid',
        'magenta', 'mediumvioletred', 'darkcyan'
    ]

    # Create a new figure with two subplots side by side
    fig, (ax_var, ax_ent) = plt.subplots(1, 2, figsize=(14, 6))

    # Determine the maximum value for variance and entropy
    min_var_values = [min(values) for key, values in test_var.items()]  # Start with min values from test_var
    max_var_values = [max(values) for key, values in test_var.items()]  # Start with max values from test_var
    min_ent_values = [min(values) for key, values in test_entropy.items()]  # Start with max values from test_entropy
    max_ent_values = [max(values) for key, values in test_entropy.items()]  # Start with max values from test_entropy

    if train_var:
        min_var_values += [min(values) for key, values in train_var.items()]
        max_var_values += [max(values) for key, values in train_var.items()]
    if train_entropy:
        min_ent_values += [min(values) for key, values in train_entropy.items()]
        max_ent_values += [max(values) for key, values in train_entropy.items()]

    min_var = min(min_var_values) if min_var_values else 0
    max_var = max(max_var_values) if max_var_values else 0
    min_ent = min(min_ent_values) if min_ent_values else 0
    max_ent = max(max_ent_values) if max_ent_values else 0

    # Plot the variance and entropy data
    for i, (label, values) in enumerate(train_var.items() if train_var else []):
        color = colors[i % len(colors)]
        ax_var.plot(end_of_epoch_step_num, values, label=f"Train {label}", linestyle='-', color=color, marker='o')
    for i, (label, values) in enumerate(test_var.items()):
        color = colors[i % len(colors)]
        ax_var.plot(end_of_epoch_step_num, values, label=f"Test {label}", linestyle='--', color=color, marker='o')

    for i, (label, values) in enumerate(train_entropy.items() if train_entropy else []):
        color = colors[i % len(colors)]
        ax_ent.plot(end_of_epoch_step_num, values, label=f"Train {label}", linestyle='-', color=color, marker='o')
    for i, (label, values) in enumerate(test_entropy.items()):
        color = colors[i % len(colors)]
        ax_ent.plot(end_of_epoch_step_num, values, label=f"Test {label}", linestyle='--', color=color, marker='o')

    # Set the titles for the subplots
    ax_var.set_title(figure_title + " - Variance")
    ax_ent.set_title(figure_title +" - Entropy")

    # Set the y-axis limits based on the maximum values found
    ax_var.set_ylim(min_var*0.95, max_var*1.05)
    ax_ent.set_ylim(min_ent*0.95, max_ent*1.05)

    # Set common x-axis and y-axis labels
    ax_var.set_xlabel("Step")
    ax_var.set_ylabel("Variance")

    ax_ent.set_xlabel("Step")
    ax_ent.set_ylabel("Entropy")

    for ax in [ax_var, ax_ent]:
        handles, labels = ax.get_legend_handles_labels()
        for key, value in features_scales.items():
            # Create a proxy artist for the new label
            handle = Line2D([], [], color='none', marker='none', linestyle='none',
                            label="{} scale = {:.3f}".format(key, value))
            handles.append(handle)
        ax.legend(handles=handles, loc='upper right')


    """
    # Gather all handles and labels from both subplots for the legend
    handles_ent, labels_ent = ax_ent.get_legend_handles_labels()
    if train_var:  # Add train_var handles and labels if train_var is provided
        handles_var, labels_var = ax_var.get_legend_handles_labels()
        handles_ent.extend(handles_var)
        labels_ent.extend(labels_var)

    # Create custom legend entries for feature scales and add them
    for key, scale in features_scales.items():
        custom_line = Line2D([0], [0], color='none', label=f"{key} scale = {scale}")
        handles_ent.append(custom_line)
        labels_ent.append("{} scale = {:.3f}".format(key, scale))

    # Place the single legend box on the entropy plot
    ax_ent.legend(handles=handles_ent, labels=labels_ent, loc='upper left', bbox_to_anchor=(1, 1))
    """

    # Set the main title for the figure
    fig.suptitle(figure_title)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rectangle to make space for the suptitle

    # Save the figure
    plt.savefig(save_path)
    plt.close()




def plot_dict(dict: Dict[str, List[float]],
              title: str,
              save_path: str,
              y_axis: str,
              x_axis: str,
              horz_dict: Dict[str, float] | None = None,
              x_list: List[float] | None = None) -> None:
    """Plots a figure of the data in the dict and saves it to the save_path.

    Args:
        dict: A dictionary where each key is a string and each value is a list of floats.
        horz_dict: drawing the values as horizontal lines
        title: The title of the plot.
        save_path: The path to save the plot.
        y_axis: The label for the y-axis.
        x_axis: The label for the x-axis.
        x_list: The x values to use for the plot.
    """

    colors = [
        'red', 'orange', 'gold', 'yellow', 'yellowgreen', 'greenyellow', 'green', 'springgreen', 'lightseagreen',
        'deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'darkblue', 'black', 'darkgrey', 'blueviolet',
        'magenta', 'pink'
    ]

    colors = [
        'red', 'orange', 'gold', 'yellowgreen', 'greenyellow', 'green', 'springgreen', 'lightseagreen',
        'deepskyblue', 'dodgerblue', 'royalblue', 'mediumblue', 'darkblue', 'black', 'darkgrey', 'blueviolet',
        'magenta', 'pink'
    ]

    if x_list == None:
        x_list = [i for i in range(len(list(dict.values())[0]))]

    min_values = [min(values) for key, values in dict.items()]
    max_values = [max(values) for key, values in dict.items()]
    min_value = min(min_values)
    max_value = max(max_values)

    fig, ax = plt.subplots()

    for i, elem in enumerate(dict.items()):
        key, values = elem

        if "test" in key:
            linestyle = "solid"
        else:
            linestyle = "dashed"

        #ax.plot(x_list, values, label=key, linestyle=linestyle, color=colors[math.floor(i/2)], marker='o')
        ax.plot(x_list, values, label=key, linestyle="solid", color=colors[i], marker='o')

    if horz_dict:
        j = -1
        for key, values in horz_dict.items():
            j += 1
            ax.hlines(y=values, xmin=x_list[0], xmax=x_list[-1],label=key, linestyle="dotted", color=colors[j])

    ax.set_title(title)
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    ax.set_ylim(min_value * 0.9, max_value * 1.1)

    # Place the legend outside the plot.
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    fig.savefig(save_path)
    plt.close(fig)


def plot_matrix_with_colorbar(matrix: np.ndarray,
                              title: str,
                              save_path: str,
                              colorbar_label: str,
                              legend_list: Optional[List[str]] = None,
                              point_list: Optional[List[Tuple[int, int]]] = None,
                              x_axis: Optional[List[float]] = None,
                              y_axis: Optional[List[float]] = None,
                              x_label: Optional[str] = None,
                              y_label: Optional[str] = None) -> None:
    """
    Plots a matrix of floats with a colorbar, title, and saves the plot to a given path.
    Marks the smallest and largest values in the matrix, and optionally custom points with their values.
    Sets x and y axis labels if provided, displaying only half of them.

    :param matrix: 2D numpy array, the matrix to be plotted.
    :param title: str, the title of the plot.
    :param save_path: str, the path to save the plot.
    :param colorbar_label: str, the label for the colorbar.
    :param legend_list: Optional list of str, labels for custom points.
    :param point_list: Optional list of tuples, coordinates (as integers) for custom points.
    :param x_axis: Optional list of floats, labels for the x-axis.
    :param y_axis: Optional list of floats, labels for the y-axis.
    :param x_label: Optional str, label for the x-axis.
    :param y_label: Optional str, label for the y-axis.
    """
    plt.figure(figsize=(10, 8))  # Set plot size
    ax = plt.gca()
    cax = ax.matshow(matrix, cmap='viridis')
    plt.title(title)

    # Function to skip every other label
    def skip_labels(labels):
        return [label if i % 2 == 0 else '' for i, label in enumerate(labels)]

    # Set x and y axis labels if provided, skipping every other one
    if x_axis is not None:
        ax.set_xticks(range(len(x_axis)))
        ax.set_xticklabels(skip_labels(x_axis))
    if y_axis is not None:
        ax.set_yticks(range(len(y_axis)))
        ax.set_yticklabels(skip_labels(y_axis))

    # Set x and y axis main labels if provided
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    # Increase font size for colorbar
    cbar = plt.colorbar(cax, label=colorbar_label, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Find smallest and largest values and their indices
    min_val, max_val = np.min(matrix), np.max(matrix)
    min_idx = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
    max_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)

    # Mark these points on the plot
    ax.scatter(min_idx[1], min_idx[0], color='blue', label=f'Min: {min_val:.2f}')
    ax.scatter(max_idx[1], max_idx[0], color='red', label=f'Max: {max_val:.2f}')

    # Plot custom points if provided
    if legend_list is not None and point_list is not None:
        for point, label in zip(point_list, legend_list):
            value = matrix[point[1], point[0]]
            ax.scatter(point[0], point[1], label=f'{label}: {value:.2f}')

    # Add legend inside the plot in the upper left corner
    ax.legend(loc='upper left')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the plot


if __name__ == '__main__':
    import scipy.stats as stats

    # P-value test
    distillation_per_task = [72.02, 57.31, 65.47, 65.03, 59.79, 79.04, 76.18, 34.55, 86.86, 74.87, 65.72, 77.04,
                             84.37, 82.62, 89.11, 87.66, 67.20, 92.39,
                             80.14, 62.93, 66.18, 82.09, 67.31, 37.83, 78.51, 90.91, 91.82]
    distillation_joint    = [66.34, 52.61, 59.64, 52.36, 49.64, 76.76, 42.67, 29.20, 75.90, 63.99, 56.22, 64.27,
                             82.20, 80.64, 86.98, 77.28, 65.84, 86.62,
                             63.28, 48.12, 63.16, 67.60, 57.67, 31.71, 75.84, 85.78, 90.58]
    fs_merge_per_task     = [71.86, 63.18, 74.59, 67.61, 62.26, 83.11, 72.93, 85.69, 91.54, 78.20, 76.13, 79.71,
                             85.82, 80.39, 93.22, 88.10, 69.23, 92.41,
                             88.46, 66.02, 73.43, 84.34, 67.43, 79.48, 78.60, 91.68, 95.77]
    fs_merge_joint        = [68.23, 59.28, 69.45, 57.98, 39.74, 80.43, 39.48, 82.10, 77.63, 65.90, 66.91, 64.94,
                             83.11, 78.43, 90.04, 79.33, 67.22, 89.74,
                             74.95, 49.42, 69.11, 71.17, 55.35, 71.24, 74.86, 90.92, 95.09]

    # Calculate the differences
    differences_per_task = [x - y for x, y in zip(fs_merge_per_task, distillation_per_task)]
    differences_joint = [x - y for x, y in zip(fs_merge_joint, distillation_joint)]

    # Perform the paired t-test
    t_statistic_per_task, p_value_per_task = stats.ttest_rel(fs_merge_per_task, distillation_per_task)
    t_statistic_joint, p_value_joint = stats.ttest_rel(fs_merge_joint, distillation_joint)

    print("per_task")
    print(f"T-statistic: {t_statistic_per_task}")
    print(f"P-value: {p_value_per_task}")
    print(f"The mean and std of the differences: {np.mean(differences_per_task):.3f} and {np.std(differences_per_task):.3f}")

    print("joint")
    print(f"T-statistic: {t_statistic_joint}")
    print(f"P-value: {p_value_joint}")
    print(f"The mean and std of the differences: {np.mean(differences_joint):.3f} and {np.std(differences_joint):.3f}")
