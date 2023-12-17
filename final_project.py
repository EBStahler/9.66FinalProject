import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from tqdm import tqdm
from itertools import product

#Open resposes. Delete rundant column
df = pd.read_csv('9.66 Final Responses Final.csv', index_col=0)
df = df.iloc[:26]
#Accidentally had a repeat picture
df.columns = [70, 10, 30, 100, 40, 80, 50, 90, 99, 60, 20, 0, 75, 15, 45, 65, 11, 55, 72, 95, 22, 5, 38, 85, 68, 35, 81, 25, 62, 47, 'drop', 17]
df = df.drop(columns=['drop'])
#print(df)

#Count number of each phrase for each question
res_dict = {'certain': [], 'very likely': [], 'likely': [], 'possible': [], 'unlikely': [], 'very unlikely': [], 'impossible': []}
for val in df.columns:
    col = df[val]
    for elt in col:
        if type(elt) != str:
            continue
        response = elt.lower()
        res_dict[response].append(val)

# Function to create the dictionary
def create_word_percentage_dict(df, words):
    """
    Create dictionary of percentage fo word use for each stiimulus probability.

    Parameters:
    - df (pandas dataFrame): Raw responses to each question
    - words (list): List of words that we're considering

    Returns:
    - dict: Dictionary mapping words to lists of percentages for each stimulus.
    """

    word_percentage_dict = {word: [] for word in words}

    for stimulus_column in df.columns:
        stimulus_responses = df[stimulus_column].values
        for i in range(len(stimulus_responses)):
            if type(stimulus_responses[i]) == str:
                stimulus_responses[i] = stimulus_responses[i].lower()

        for word in words:
            # Count the occurrences of the word for the current stimulus
            word_count = np.count_nonzero(stimulus_responses == word)
            total_responses = len(stimulus_responses)

            # Calculate the percentage and append to the dictionary
            percentage = word_count / total_responses if total_responses > 0 else 0.0
            word_percentage_dict[word].append(percentage)

    return word_percentage_dict

#Estimate threshold based responses
def generate_model_responses(stimuli, words, monotonicity_types, thresholds):
    """
    Generate model responses based on threshold values.

    Parameters:
    - stimuli (numpy.ndarray): Array of values representing true probabilities for each stimulus.
    - words (list): List of words.
    - monotonicity_types (dict): Dictionary specifying monotonicity type for each word.
    - thresholds (numpy.ndarray): Threshold values for each word.

    Returns:
    - dict: Dictionary mapping words to lists of percentages for each stimulus.
    """
    model_responses = {word: [] for word in words}

    for stimulus in stimuli:
        for word, threshold in zip(words, thresholds):
            if monotonicity_types[word] == 'increasing' and stimulus >= threshold:
                # Calculate probability proportional to the distance from the threshold
                probability = (1)/(stimulus-threshold+0.01)**1#(stimulus - threshold) / (1.0 - threshold)
                model_responses[word].append(probability)
            elif monotonicity_types[word] == 'decreasing' and stimulus <= threshold:
                # Calculate probability proportional to the distance from the threshold
                probability = (1)/(threshold-stimulus+0.01)**1#(threshold - stimulus) / threshold
                model_responses[word].append(probability)
            else:
                model_responses[word].append(0.0)  # Word is not true for the stimulus

        # Normalize percentages for each stimulus
        total_probability = sum(model_responses[word][-1] for word in words)
        for word in words:
            model_responses[word][-1] /= total_probability if total_probability != 0 else 1.0

    return model_responses

#Estimate prototype based responses
def generate_model_responses_prototype(stimuli, words, prototypes, distances):
    """
    Generate model responses based on prototype values and distance measures.

    Parameters:
    - stimuli (numpy.ndarray): Array of values representing true probabilities for each stimulus.
    - words (list): List of words.
    - prototypes (numpy.ndarray): Prototype values for each word.
    - distances (numpy.ndarray): Distance measures for each word.

    Returns:
    - dict: Dictionary mapping words to lists of percentages for each stimulus.
    """
    model_responses = {word: [] for word in words}

    for stimulus in stimuli:
        for word, prototype, distance in zip(words, prototypes, distances):
            probability = np.exp(-((stimulus - prototype) / distance)**2)
            model_responses[word].append(probability)

        # Normalize percentages for each stimulus
        total_probability = sum(model_responses[word][-1] for word in words)
        for word in words:
            model_responses[word][-1] /= total_probability if total_probability != 0 else 1.0

    return model_responses

#KL divergence of two distributions
def kl_divergence(p, q, epsilon=1e-10):
    """
    Calculate the KL Divergence between two probability distributions.

    Parameters:
    - p (numpy.ndarray): Probability distribution 1.
    - q (numpy.ndarray): Probability distribution 2.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - float: KL Divergence.
    """
    p = np.array(p)
    q = np.array(q)

    # Ensure probabilities sum to 1 (normalize if necessary)
    p /= np.sum(p)
    q /= np.sum(q)

    # Add epsilon to avoid division by zero
    p += epsilon
    q += epsilon

    return np.sum(p * np.log(p / q))

#Sum of KL divergence over all words
def total_kl_divergence(data_probs, model_probs, epsilon=1e-10):
    """
    Calculate the total KL Divergence between data and model probabilities for each word.

    Parameters:
    - data_probs (dict): Dictionary mapping words to lists of probabilities from the data.
    - model_probs (dict): Dictionary mapping words to lists of probabilities from the model.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - float: Total KL Divergence.
    """
    total_kl = 0.0

    for word in data_probs:
        p = np.array(data_probs[word])
        q = np.array(model_probs[word])

        total_kl += kl_divergence(p, q, epsilon)

    return total_kl

#Attempt to optimize thresholds
def optimize_thresholds(data_responses, words, stimuli, monotonicity_types, initial_thresholds):
    """
    Optimize threshold values to minimize the total KL Divergence between data and model responses.

    Parameters:
    - data_responses (dict): Dictionary mapping words to lists of probabilities from the data.
    - words (list): List of words.
    - stimuli (numpy.ndarray): Array of values representing true probabilities for each stimulus.
    - monotonicity_types (dict): Dictionary specifying monotonicity type for each word.
    - initial_thresholds (numpy.ndarray): Initial threshold values for each word.

    Returns:
    - numpy.ndarray: Optimized threshold values.
    """
    # Define the objective function to minimize (total KL Divergence)
    def objective_function(thresholds):
        model_responses = generate_model_responses(stimuli, words, monotonicity_types, thresholds)
        total_kl = total_kl_divergence(data_responses, model_responses)
        return total_kl

    # Perform the optimization
    result = minimize(objective_function, initial_thresholds, method='L-BFGS-B')

    return result.x

#Attempt to optimize prototypes and distances
def optimize_prototypes_distances(data_responses, words, stimuli, initial_prototypes, initial_distances):
    """
    Optimize prototype values and distances to minimize the total KL Divergence between data and model responses.

    Parameters:
    - data_responses (dict): Dictionary mapping words to lists of probabilities from the data.
    - words (list): List of words.
    - stimuli (numpy.ndarray): Array of values representing true probabilities for each stimulus.
    - initial_prototypes (numpy.ndarray): Initial prototype values for each word.
    - initial_distances (numpy.ndarray): Initial distance measures for each word.

    Returns:
    - tuple: Optimized prototype values and distances.
    """
    # Define the objective function to minimize (total KL Divergence)
    def objective_function(params):
        num_words = len(words)
        prototypes = params[:num_words]
        distances = params[num_words:]

        model_responses = generate_model_responses_prototype(stimuli, words, prototypes, distances)
        total_kl = total_kl_divergence(data_responses, model_responses)
        return total_kl

    # Combine initial prototypes and distances into a single array
    initial_params = np.concatenate([initial_prototypes, initial_distances])

    # Perform the optimization
    result = minimize(objective_function, initial_params, method='L-BFGS-B')

    optimized_prototypes = result.x[:len(words)]
    optimized_distances = result.x[len(words):]
    return optimized_prototypes, optimized_distances

#Make a histogram-like plot of responses 
def plot_from_percent_dict(stimuli, word_percentage_dict, bin_edges = np.arange(0,110,5)):
    # Create an array to store the histogram counts for each word and bin
    histogram_counts = np.zeros((len(word_percentage_dict), len(bin_edges) - 1))

    # Calculate the histogram for each word
    for idx, (word, percentages) in enumerate(word_percentage_dict.items()):
        histogram, _ = np.histogram(stimuli, bins=bin_edges, weights=percentages)
        histogram_counts[idx, :] = histogram

    # Normalize the counts to percentages
    histogram_percentages = histogram_counts / np.sum(histogram_counts, axis=0) * 100

    # Create a stacked bar plot
    fig, ax = plt.subplots()

    bottom = np.zeros(len(bin_edges) - 1)
    for word_idx, (word, _) in enumerate(word_percentage_dict.items()):
        ax.bar(bin_edges[:-1], histogram_percentages[word_idx, :], width=np.diff(bin_edges), label=word, bottom=bottom)
        bottom += histogram_percentages[word_idx, :]

    # Add labels and legend. Change title depending on the plot
    ax.set_xlabel('True Stimulus Value')
    ax.set_ylabel('Percent Response Rate')
    ax.set_title('Optimal Prototype Based Lexicon Prediction')
    ax.legend()
    fig.set_figwidth(15)

    # Show the plot
    plt.show()

#Make density plot of participant responses
def make_density_plot(res_dict):
    # Define a color palette for the plots
    colors = sns.color_palette('husl', n_colors=len(res_dict))

    # Create subplots for each word's usage
    num_words = len(res_dict)
    fig, axes = plt.subplots(num_words, 1, figsize=(12, 3.5), sharex=True)

    # Plot density for each word with different colors
    for (word, stimulus_list), ax, color in zip(res_dict.items(), axes, colors):
        sns.kdeplot(stimulus_list, ax=ax, fill=True, color=color, common_norm=False)
        ax.set_xlabel('Stimulus Value')
        ax.set_ylabel('')  # Remove y-axis label
        ax.yaxis.set_ticks([])  # Remove y-axis ticks
        ax.text(-0.1, 0.5, word, transform=ax.transAxes, va='center', ha='right', fontweight='bold', color='black')

        # Calculate and display mean, 5th, and 95th percentiles
        mean_value = np.mean(stimulus_list)
        percentile_5 = np.percentile(stimulus_list, 5)
        percentile_95 = np.percentile(stimulus_list, 95)
        
        ax.text(1.1, 0.5, f'Mean: {mean_value:.2f}\n5%-95% Percentiles: ({percentile_5:.2f}, {percentile_95:.2f})',
                transform=ax.transAxes, va='center', ha='left', fontsize=8, color='black')
                
        ax.set_xlim(0, 100)

    # Add a common title
    fig.suptitle('Density Plot of Word Usage Across Stimuli', fontsize=14)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    stimuli = df.columns 
    prec_stimuli = df.columns/100
    words = ['impossible', 'very unlikely', 'unlikely', 'possible', 'likely', 'very likely', 'certain']
    monotonicity_types = {
        'impossible': 'decreasing',
        'very unlikely': 'decreasing',
        'unlikely': 'decreasing',
        'possible': 'increasing',
        'likely': 'increasing',
        'very likely': 'increasing',
        'certain': 'increasing',
    }

    #My results
    res_word_percentage_dict = create_word_percentage_dict(df, words)
    plot_from_percent_dict(stimuli, res_word_percentage_dict)

    #opt thresh that the for loops generated: [0, 0.4, 0.6, 0.2, 0.3, 0.4, 1]
    #Before using itertools
    '''opt_thresholds = [0, 0.4, 0.6, 0.2, 0.3, 0.4, 1]
    opt_thresh_perc_dict = generate_model_responses(prec_stimuli, words, monotonicity_types, opt_thresholds)
    plot_from_percent_dict(stimuli, opt_thresh_perc_dict)
    print(total_kl_divergence(res_word_percentage_dict, opt_thresh_perc_dict))'''

    '''    min_loss = np.inf
    opt_thresh = [0,0,0,0,0,0,0]
    for imp_thresh in tqdm(np.arange(0, 105, 10)):
        for vu_thresh in np.arange(0, 105, 10):
            for u_thresh in tqdm(np.arange(0, 105, 10)):
                for p_thresh in tqdm(np.arange(0, 105, 10)):
                    for l_thresh in np.arange(0, 105, 10):
                        for vl_thresh in np.arange(0, 105, 10):
                            for c_thresh in np.arange(0, 105, 10):
                                threshold = [imp_thresh, vu_thresh, u_thresh, p_thresh, l_thresh, vl_thresh, c_thresh]
                                thresh_prec_dict = generate_model_responses(stimuli, words, monotonicity_types, threshold)
                                loss = total_kl_divergence(res_word_percentage_dict, thresh_prec_dict)
                                if loss < min_loss:
                                    min_loss = loss
                                    opt_thresh = threshold
    
    print(opt_thresh)
    print(min_loss)'''

    '''threshold_ranges = [
        np.arange(0, 11, 2)/100,  # Range for the first threshold
        np.arange(30, 51, 2)/100,    # Range for the second threshold
        np.arange(50, 71, 2)/100,  # Range for the third threshold
        np.arange(10, 31, 2)/100,
        np.arange(20, 41, 2)/100,
        np.arange(30, 51, 2)/100,
        np.arange(90, 101, 2)/100
    # ... Add ranges for the other thresholds
    ]

    threshold_combinations = list(product(*threshold_ranges))

    min_loss = np.inf
    opt_thresh = [0] * len(words)

    for thresholds in tqdm(threshold_combinations):
        thresh_prec_dict = generate_model_responses(prec_stimuli, words, monotonicity_types, thresholds)
        loss = total_kl_divergence(res_word_percentage_dict, thresh_prec_dict)

        if loss < min_loss:
            min_loss = loss
            opt_thresh = thresholds

    print("Optimized Thresholds:", opt_thresh)
    print('Min loss:', min_loss)'''

    #Opt thresholds generated by loop above:
    opt_thresh = [0.0, 0.4, 0.56, 0.12, 0.26, 0.4, 1.0]
    word_prec_dict = generate_model_responses(prec_stimuli, words, monotonicity_types, opt_thresh)
    plot_from_percent_dict(stimuli, word_prec_dict)
    print(total_kl_divergence(res_word_percentage_dict, word_prec_dict))

    '''
    RESULTS of optimiizing prototypes and distances

    protos: [0.0, 0.11, 0.27, 0.48, 0.7, 0.8, 1.0]
    dists: [0.02, 0.1, 0.14, 0.2, 0.16, 0.16, 0.02]
    loss: 1.446

    opt_protos = [0, 0.14, 0.32, 0.48, 0.65, 0.85, 1]
    opt_distances = [0.01, 0.12, 0.15, 0.18, 0.15, 0.1, 0.01]
    loss: 1.746

    protos: [0.0, 0.15, 0.35, 0.46, 0.71, 0.89, 1.0]
    dists: [0.05, 0.15, 0.15, 0.15, 0.15, 0.15, 0.05]
    loss: 2.4828163249756243

    protos:[0.0, 0.15, 0.35, 0.46, 0.71, 0.89, 1.0]
    dists: (0.02, 0.08, 0.16, 0.16, 0.16, 0.2, 0.02)
    loss: 1.498

    protos: (0.0, 0.12, 0.3, 0.47, 0.7, 0.91, 1)
    dists: (0.02, 0.08, 0.16, 0.16, 0.16, 0.2, 0.02)
    loss: 1.317

    protos: (0.0, 0.12, 0.3, 0.47, 0.7, 0.91, 1)
    dists: (0.005, 0.08, 0.16, 0.18, 0.14, 0.22, 0.005)
    loss: 0.671

    protos: (0.0, 0.115, 0.275, 0.47, 0.7, 0.925, 1.0000000000000002)
    dists: (0.005, 0.08, 0.16, 0.18, 0.14, 0.22, 0.005)
    loss: 0.654
    '''

    opt_protos = [0.0, 0.115, 0.275, 0.47, 0.7, 0.925, 1]
    prototype_ranges = [
        np.arange(0,1.1,0.2)/100,
        np.arange(9,12,0.5)/100,
        np.arange(27,32,0.5)/100,
        np.arange(45,49,0.5)/100,
        np.arange(68,72,0.5)/100,
        np.arange(88,94,0.5)/100,
        np.arange(99,100.1,0.2)/100
    ]

    opt_distances = [0.005, 0.08, 0.16, 0.18, 0.14, 0.22, 0.005]
    distance_ranges = [
        np.arange(0,0.02,0.005),
        np.arange(0.04,0.11,0.02),
        np.arange(0.12,0.21,0.02),
        np.arange(0.12,0.21,0.02),
        np.arange(0.12,0.21,0.02),
        np.arange(0.13,0.25,0.03),
        np.arange(0,0.02,0.005),
    ]

    min_loss = np.inf
    opt_proto = [0] * len(words)
    opt_dist = [0] * len(words)

    prototype_combinations = list(product(*prototype_ranges))
    distance_combinations = list(product(*distance_ranges))

    '''for prototype in tqdm(prototype_combinations):
        prod_prec_dict = generate_model_responses_prototype(prec_stimuli, words, prototype, opt_distances)
        loss = total_kl_divergence(res_word_percentage_dict, prod_prec_dict)

        if loss < min_loss:
            min_loss = loss
            opt_proto = prototype

    print("Optimized Thresholds:", opt_proto)
    print('Min loss:', min_loss)'''

    #Plot optimal prototypes/distances
    pred_dict = generate_model_responses_prototype(prec_stimuli, words, opt_protos, opt_distances)
    print(total_kl_divergence(res_word_percentage_dict, pred_dict))
    plot_from_percent_dict(stimuli, pred_dict)

    #Make density plot
    res_dict_no_edges = {word: res_dict[word] for word in res_dict if (word not in ['impossible', 'certain'])}
    make_density_plot(res_dict_no_edges)