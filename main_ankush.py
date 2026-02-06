# %% [markdown]
# # Streamlined ML Pipeline: K-means Split → Random Forest → Connected Components

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from itertools import product
from tqdm.auto import tqdm
import csv
import time
import datetime
from sklearn.inspection import PartialDependenceDisplay
# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from scipy.stats import spearmanr, pearsonr, ks_2samp, percentileofscore


# %% [markdown]
# # Helper Functions``

# %%
def elbow_method(df, headers, max_clusters=20, directory_figure='Figures'):
    """Generate elbow plot for K-means clustering"""
    input_df = df[headers].values
    scaler = StandardScaler()
    normalized_input_df = scaler.fit_transform(input_df)
    
    inertia = []
    range_clusters = range(1, max_clusters+1) 
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(normalized_input_df)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(6, 6))
    plt.plot(range_clusters, inertia, marker='o', markersize=5, color='black')
    plt.xlabel('Number of Clusters', fontsize=24)
    plt.ylabel('WCSS', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xticks(np.arange(0, max_clusters+1, 4))
    
    os.makedirs(directory_figure, exist_ok=True)
    filepath = os.path.join(directory_figure, 'kmeans_elbow.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return normalized_input_df

def generate_train_test_indices(num_clusters, normalized_input_df, samples_per_cluster=5, random_state=42):
    """Generate train/test indices using K-means clustering"""
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_input_df)


    test_indices = []
    for cluster in np.unique(cluster_labels):
        cluster_points = np.where(cluster_labels == cluster)[0]
        rng = np.random.RandomState(random_state)
        representative_indices = rng.choice(cluster_points, size=samples_per_cluster, replace=False)
        test_indices.extend(representative_indices)

    test_indices = np.array(test_indices)
    train_indices = np.setdiff1d(np.arange(len(normalized_input_df)), test_indices)

    return train_indices, test_indices

def perform_ks_test(train_indices, test_indices, df, columns, directory_data="DataExport"):
    """Perform Kolmogorov-Smirnov test for train/test distribution similarity"""
    ks_results = {}
    
    for column in columns:
        train_data = df[column].iloc[train_indices]
        test_data = df[column].iloc[test_indices]
        ks_statistic, p_value = ks_2samp(train_data, test_data)
        ks_results[column] = {'KS Statistic': ks_statistic, 'p-value': p_value}
    
    ks_df = pd.DataFrame.from_dict(ks_results, orient="index")
    os.makedirs(directory_data, exist_ok=True)
    file_path = os.path.join(directory_data, 'kmeans_kstest.csv')
    ks_df.to_csv(file_path)
    
    print(f"KS test results saved to: {file_path}")
    return ks_results

def evaluate_model(y_test, y_pred, model_name,directory='DataExport',logfile='modeloutput.csv'):
    """Evaluate model performance"""
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred)[0]
    pearson = pearsonr(y_test, y_pred)[0]
    
    print(f"\n{model_name} Performance:")
    print(f"  MSE: {mse:.3f}")
    print(f"  Explained Variance: {evs:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Spearman: {spearman:.3f}")
    print(f"  Pearson: {pearson:.3f}")
    
    headers = ['Model','Mean Squared Error','Explained Variance','R^2','Timestamp','Spearman','Pearson']
    #filename = 'modeloutput.csv'
    file_path = os.path.join(directory, logfile)
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Determine whether to write headers: file should not exist or be empty
    write_headers = not os.path.exists(file_path) or os.stat(file_path).st_size == 0    
    
    # Open the file in append mode (creates the file if it does not exist)
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if necessary
        if write_headers:
            writer.writerow(headers)
        # Write the data
        writer.writerow([model_name,
                         np.round(mean_squared_error(y_test, y_pred), 3),
                         np.round(explained_variance_score(y_test, y_pred), 3),
                         np.round(r2_score(y_test, y_pred), 3),
                         current_datetime,
                         np.round(spearmanr(y_test, y_pred), 3),
                         np.round(pearsonr(y_test, y_pred), 3)
                         ])
        
    return np.round(r2_score(y_test, y_pred), 3)
    
    return r2

def find_connected_components(grid):
    """Find connected components using iterative depth-first search"""
    grid_shape = grid.shape
    num_dims = len(grid_shape)
    visited = np.zeros(grid_shape, dtype=bool)
    components = []
    components_indices = []
    
    neighbor_offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * num_dims)).T.reshape(-1, num_dims)
    neighbor_offsets = neighbor_offsets[np.any(neighbor_offsets, axis=1)]
    
    def get_neighbors(position):
        neighbors = [tuple(np.array(position) + offset) for offset in neighbor_offsets]
        return [pos for pos in neighbors if all(0 <= pos[i] < grid_shape[i] for i in range(num_dims))]

    total = np.prod(grid_shape)
    total_component = np.count_nonzero(grid == 1)
    pbar = tqdm(total=total_component, desc="Scanning Components")
    for position in np.ndindex(grid_shape):
        if (grid[position] == 1) and (not visited[position]):
            stack = [position]
            component = np.zeros(grid_shape, dtype=bool)
            component_index = []

            while stack:
                current_pos = stack.pop()
                before = np.count_nonzero(visited)
                if not visited[current_pos]:
                    visited[current_pos] = True
                    component[current_pos] = True
                    after = np.count_nonzero(visited)
                    pbar.update(after - before)
                    component_index.append(current_pos)
                    for neighbor in get_neighbors(current_pos):
                        if (grid[neighbor] == 1) and (not visited[neighbor]):
                            stack.append(neighbor)
            
            components.append(component)
            components_indices.append(component_index)

    return components, components_indices

def convert_to_actual_values(normalized_indices, l_limit, u_limit):
    """Convert normalized indices to actual parameter values"""
    return l_limit + (normalized_indices * (u_limit - l_limit))

def calculate_fill_factor(component_indices, y_pred, shape, threshold, pce_max, progress_callback=None):
    """Calculate fill factor (robustness metric) for a component"""
    fraction_total = 0
    
    for index in component_indices:
        pce = y_pred[np.ravel_multi_index(index, shape)]
        
        fraction_point = 0
        if (pce_max - threshold) > 0:
            fraction_point = (pce - threshold) / (pce_max - threshold)
        
        if (pce - threshold) >= 0:
            fraction_total += fraction_point
        if progress_callback is not None:
            progress_callback(1)
    
    if len(component_indices) == 0:
        return 0
    
    return fraction_total / len(component_indices)


def analyze_components(components_indices, y_pred, x_test, shape, threshold, l_limit, u_limit):
    """Analyze connected components and extract statistics"""
    components_info = []
    def update_progress(n):
        pbar.update(n)

    total_points = sum(len(c) for c in components_indices)
    pbar = tqdm(total=total_points, desc="Analyzing components")


    for component_num, component_indices in enumerate(components_indices, 1):
        pce_values = [y_pred[np.ravel_multi_index(idx, shape)] for idx in component_indices]
        
        # Filter by threshold
        pce_above_threshold = [p for p in pce_values if p >= threshold]
        
        if len(pce_above_threshold) == 0:
            continue
            
        max_pce = max(pce_values)
        mean_pce = np.mean(pce_above_threshold)
        num_points = len(pce_above_threshold)

        # percentiles
        stats_p16 = np.percentile(pce_above_threshold, 16)
        stats_p84 = np.percentile(pce_above_threshold, 84)
        
        # Find max PCE location
        max_pce_index = component_indices[np.argmax(pce_values)]
        normalized_max_pce_index = x_test[np.ravel_multi_index(max_pce_index, shape)]
        
        # Calculate fill factor
        fill_factor = calculate_fill_factor(component_indices, y_pred, shape, threshold, max_pce,progress_callback=update_progress)

        #Fallback values for nonexistent components
        if num_points == 0:
            num_points = 0
            mean_pce = max_pce
            fill_factor = 0
            stats_p16 = max_pce
            stats_p84 = max_pce
        
        components_info.append({
            "component_number": component_num,
            "num_points": num_points,
            "mean_pce": mean_pce,
            "max_pce": max_pce,
            "fill_factor": fill_factor,
            "stats_p16": stats_p16,
            "stats_p84": stats_p84,
            "max_pce_coordinates": normalized_max_pce_index
        })
    pbar.close()
    return components_info

def save_to_csv(filename, data, thresholds, directory_data):
    """Save component data to CSV file"""
    filepath = os.path.join(directory_data, filename)
    headers = ["Threshold"] + [f"Component {i+1}" for i in range(len(data))]

    # Transpose data so each row corresponds to a threshold
    rows = list(zip(thresholds, *data))

    # Write data to CSV
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"CSV file '{filepath}' has been created successfully!")

def opvpseudobag(df,nth=1,randomstate=42,ind=['set','donor_ratio','concentration','annealing_t','spinspeed','acetone_v_perc','status','filter_trial'],dep=['pce','voc','jsc','ff'], 
             name="bagged_data.csv",exporttocsv=True,decimal=4,filter=True,path="DataExport"):
    #Variable Definitions: 
        #independent --> an array of all column titles that are CONSISTENT between all versions of the sample, eg processing conditions or sample #
        #dependent --> Also and array of column headers, this time focusing on the columns to be averaged

    #independent=['set','donor_ratio','concentration','annealing_T','spinspeed','acetone_v_perc'] #For example
    #dependent = ['pce','voc','jsc','ff']
    if filter == True:
        df = df[df['status'].str.lower()=='accept']
        df = df[df['filter_trial'].str.lower()=='accept']

    #bootdf = df.groupby(ind).nth(nth).reset_index()
    bootdf = (
        df
        .groupby(ind, group_keys=False)
        .apply(lambda g: g.iloc[nth % len(g)])
        .reset_index(drop=True)
    )

    bootdf.head()
    if exporttocsv == True: bootdf.to_csv(name, index=False)

    if exporttocsv:
        # Ensure filename ends with .csv
        if not name.endswith(".csv"):
            name = name + ".csv"

        # Convert path to Path object (no trailing slash required)
        outdir = Path(path)
        outdir.mkdir(parents=True, exist_ok=True)

        bootdf.to_csv(outdir / name, index=False)

    return bootdf

def avgsample (df,ind=['donor_ratio','concentration','annealing_t','spinspeed','acetone_v_perc','status','filter_trial'],dep=['pce','voc','jsc','ff'], 
             name="averaged_data.csv",exporttocsv=True,decimal=4,filter=True,path="DataExport/",id = 'set'):
    #Variable Definitions: 
        #independent --> an array of all column titles that are CONSISTENT between all versions of the sample, eg processing conditions or sample #
        #dependent --> Also and array of column headers, this time focusing on the columns to be averaged

    #independent=['set','donor_ratio','concentration','annealing_T','spinspeed','acetone_v_perc'] #For example
    #dependent = ['pce','voc','jsc','ff']
    if filter == True:
        df = df[df['status'].str.lower()=='accept']
        df = df[df['filter_trial'].str.lower()=='accept']

    #avgdf = df.groupby(ind)[dep].mean(numeric_only=True).round(decimal).reset_index() #Added code for averaging by sample
    avgdf = df.groupby(ind).mean(numeric_only=True).round(decimal).reset_index() #New Version to accept all outputs
    avgdf = avgdf.sort_values(by=id)    
    if exporttocsv == True: avgdf.to_csv(name, index=False)

    if exporttocsv==True:
        if name[-4:] != ".csv":
            name = name+".csv"
        if not (path[-1] == "/"):
            raise Exception("Remember to include a trailing slash on the folder directory")
        filepath = Path(path)
        filepath.mkdir(parents=True, exist_ok=True)
        avgdf.to_csv(filepath/name, index = False)
        print(f"[avgsample] Averaged data exported to: {filepath/name}")

    return avgdf


# %% [markdown]
# # Configuration

# %%
# Paths
filename = 'DOE_Ace_avg_keep.csv'  # Update with your filename
base_name, ext = os.path.splitext(filename)
directory_figure = os.path.join(base_name, 'Figures')
directory_data = os.path.join(base_name, 'DataExport')
#filename = 'DOE_Ace_avg.csv'  # Update with your filename
baggingname = filename[:-4]+'_bag'#suffix or name format to use for bagged file versions

#Bagging Parameters

# Parameters
input_headers = ['donor_ratio', 'concentration', 'spinspeed', 'annealing_t', 'sol_add_v_perc']
output_header = ['pce']
random_state = 42
test_size = 0.2
total_bags = 5 
bag = None #integer --> Default is None or integer to refer to the chosen bag number


# Parameter bounds for prediction grid
l_limit = np.array([0.5, 8.0, 800, 55, 0.0])
u_limit = np.array([1.5, 22, 6000, 130, 5.0])

# Grid resolution
resolution = 10 #Default: 20
threshold_pce = 9  # User Input on minimum PCE threshold, recommend 9
number_of_thresholds = 4 # Number of thresholds to analyze between initial threshold and max PCE, default 16

# %% [markdown]
# # Load Data

# %%
df = pd.read_csv(filename)
df = df[df['status'].str.lower() == 'accept']
df = df[df['filter_trial'].str.lower() == 'accept']
print(f"Loaded {len(df)} samples")

# %% Breakdown and PseudoBagging OR averaging
if bag is not None:
    for n in range(total_bags):
        #place data in the data export folder
        opvpseudobag(df,nth=n,ind=input_headers,dep=output_header,name=baggingname+"_"+str(n),path=directory_data) 

    filename = filename[:-4] +"_bag_" + str(bag) + ".csv"
    df = pd.read_csv(os.path.join(directory_data,filename)) #Trial Comparison

    directory_figure = os.path.join(directory_figure,str(bag))
    directory_data =os.path.join(directory_data,str(bag))


    df = df[df['status'].str.lower()=='accept']
    df = df[df['filter_trial'].str.lower()=='accept']

    # Create the directory_data if it doesn't exist
    if not os.path.exists(directory_data):
        os.makedirs(directory_data)
    # Create the directory_data if it doesn't exist
    if not os.path.exists(directory_figure):
        os.makedirs(directory_figure)

    print("Data Directory: "+directory_data)
    print("Figure Directory: "+directory_figure)
    print(filename + " Will be imported")

else: 
    avgname = filename[:-4]+'_avg.csv'
    df_avg = avgsample(df,ind=input_headers,dep=output_header,name=avgname,id='set')
    print(f"Averaged down to {len(df_avg)} samples")
    df = df_avg



# %% [markdown]
# # K-means Train-Test Split

# %%
headers = input_headers + output_header
normalized_df = elbow_method(df=df, headers=headers, max_clusters=30, directory_figure=directory_figure)
# %%
# Choose number of clusters based on elbow plot
num_clusters = 8 # User-defined based on elbow plot
samples_per_cluster = int((test_size * normalized_df.shape[0]) // num_clusters)
print(f"Samples per cluster: {samples_per_cluster}")

train_indices, test_indices = generate_train_test_indices(
    num_clusters=num_clusters, 
    normalized_input_df=normalized_df, 
    random_state=random_state, 
    samples_per_cluster=samples_per_cluster
)
print(f"Train Indices: {train_indices}")
print(f"Test Indices: {test_indices}")
# Perform KS test
ks_results = perform_ks_test(train_indices, test_indices, df, headers, directory_data=directory_data)

# %%
# Create train/test sets
X_train = df[input_headers].iloc[train_indices]
X_test = df[input_headers].iloc[test_indices]
y_train = df[output_header].iloc[train_indices].values.ravel()
y_test = df[output_header].iloc[test_indices].values.ravel()

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# %% [markdown]
# # Random Forest Model



# Train Random Forest with StandardScaler pipeline
rfA_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=random_state))
])

rfA_pipe.fit(X_train, y_train)

# Predict and evaluate
y_pred = rfA_pipe.predict(X_test)
name = "Random Forest Pipeline"
if (bag is not None): 
    name = "Random Forest Pipeline (Bag "+str(bag)+")"
r2 = evaluate_model(y_test, y_pred, name)

# Retrain on full dataset for predictions
X_full = df[input_headers]
y_full = df[output_header].values.ravel()
rfA_pipe.fit(X_full, y_full)

# %% Create Feature importance plot
# Feature importance on default RF model
rf_model = rfA_pipe.named_steps['rf']
start_time = time.time()
importances_rfA = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
elapsed_time = time.time() - start_time


forest_importances = pd.Series(importances_rfA, index=input_headers)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
fig.savefig(os.path.join(directory_figure, "feature_importance_rf.png"), dpi=300, bbox_inches='tight')

# %% Partial Dependence Plots

def axis_label(var):
    entry = label_map.get(var)
    if entry is None:
        return var
    if entry["unit"]:
        return f"{entry['name']} ({entry['unit']})"
    return entry["name"]


def title_label(var):
    entry = label_map.get(var)
    if entry is None:
        return var
    return entry["name"]


label_map = {
    "spinspeed": {
        "name": "Spin Speed",
        "unit": "rpm"
    },
    "sol_add_v_perc": {
        "name": "Solvent Additive",
        "unit": "v/v %"
    },
    "concentration": {
        "name": "Concentration",
        "unit": "mg/mL"
    },
    "annealing_t": {
        "name": "Annealing Temperature",
        "unit": "°C"
    },
    "donor_ratio": {
        "name": "Donor Ratio",
        "unit": None
    }
}


#1D Partial Dependence Plots
feature_list = X_full.columns.tolist()

for feature in feature_list:
    fig, ax = plt.subplots(figsize=(5, 4))


    disp=PartialDependenceDisplay.from_estimator(
        estimator=rfA_pipe,
        X=X_full,
        features=[feature],
        kind='average',
        grid_resolution=25,
        ax=ax
    )
    # --- IMPORTANT: override labels AFTER PDP creation ---
    disp.axes_[0, 0].set_xlabel(axis_label(feature))
    disp.axes_[0, 0].set_title(f"PDP: {title_label(feature)}")

    plt.tight_layout()
    filepath = os.path.join(directory_figure,'PDP1D_'+feature)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.close()


#2D Partial Dependence Plots
feature_pairs = [
    ('donor_ratio', 'spinspeed'),
    ('concentration', 'spinspeed'),
    ('donor_ratio','concentration'),
    ('annealing_t', 'sol_add_v_perc'),
    ('donor_ratio', 'sol_add_v_perc'),
    ('spinspeed','annealing_t'),
    ('annealing_t','concentration')
]

for f1, f2 in feature_pairs:

    # --- Create new figure for each PDP ---
    fig, ax = plt.subplots(figsize=(5, 4))

    disp=PartialDependenceDisplay.from_estimator(
        estimator=rfA_pipe,
        X=X_full,
        features=[(f1, f2)],
        kind='average',
        grid_resolution=25,
        ax=ax
    )

    # --- IMPORTANT: override labels AFTER PDP creation ---
    disp.axes_[0, 0].set_xlabel(axis_label(f1))
    disp.axes_[0, 0].set_ylabel(axis_label(f2))

    # --- Titles and labels ---
    ax.set_title(f"2D Partial Dependence: {title_label(f1)} vs {title_label(f2)}")

    plt.tight_layout()

    filepath = os.path.join(directory_figure,'PDP2D_'+f1+"_"+f2)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
# %% [markdown]
# # Generate Prediction Grid

# %%
# Create normalized grid
grid_points = [np.linspace(0, 1, resolution) for _ in range(5)]
parameter_space = list(product(*grid_points))
x_test_normalized = np.array(parameter_space)

# Convert to actual values
x_test_actual = x_test_normalized * (u_limit - l_limit) + l_limit
df_x_test = pd.DataFrame(x_test_actual, columns=input_headers)

# Predict
y_pred_grid = rfA_pipe.predict(df_x_test)

print(f"Generated {len(y_pred_grid)} predictions")
print(f"PCE range: {y_pred_grid.min():.2f} - {y_pred_grid.max():.2f}")

# %% [markdown]
# # Connected Components Analysis

# %%
# Set threshold

percentile = percentileofscore(y_pred_grid, threshold_pce)
print(f"Threshold: {threshold_pce:.2f}% PCE ({percentile:.1f}th percentile)")

# Create binary grid
grid = (y_pred_grid > threshold_pce).astype(int)
grid_shape = tuple([resolution] * 5)
grid = grid.reshape(grid_shape)

# Count points above threshold
count_above = np.sum(grid)
print(f"Points above threshold: {count_above} ({100*count_above/len(y_pred_grid):.2f}%)")

# %%
# Find connected components
components, components_indices = find_connected_components(grid)
print(f"Found {len(components)} connected components")

# %%
# Analyze components
components_info = analyze_components(
    components_indices, y_pred_grid, x_test_normalized, 
    grid_shape, threshold_pce, l_limit, u_limit
)

# Display results
for comp in components_info:
    actual_coords = convert_to_actual_values(comp['max_pce_coordinates'], l_limit, u_limit)
    print(f"\nComponent {comp['component_number']}:")
    print(f"  Points: {comp['num_points']}")
    print(f"  Mean PCE: {comp['mean_pce']:.2f}%")
    print(f"  Max PCE: {comp['max_pce']:.2f}%")
    print(f"  Fill Factor: {comp['fill_factor']:.2f}")
    print(f"  16th Percentile PCE: {comp['stats_p16']:.2f}%")
    print(f"  84th Percentile PCE: {comp['stats_p84']:.2f}%")
    print(f"  Max location: {', '.join(f'{x:.2f}' for x in actual_coords)}")

# %%
# Export results
os.makedirs(directory_data, exist_ok=True)
results_df = pd.DataFrame(components_info)
results_df.to_csv(os.path.join(directory_data, 'components_analysis.csv'), index=False)
print(f"\nResults saved to {directory_data}/components_analysis.csv")

# %%
# Create array of thresholds from initial threshold to max PCE

persistence_thresholds = np.linspace(start=threshold_pce, stop=y_pred_grid.max(), num=number_of_thresholds, retstep=False)
components_info_plot = [[] for _ in range(number_of_thresholds)]

#UPDATED LOOP
for index, value in tqdm(enumerate(persistence_thresholds),desc="Iterating Thresholds", total=len(persistence_thresholds)):
    for component_num, component_indices in enumerate(components_indices, 1):
        # Collect information for the component
        components_info_plot[index] = analyze_components(
            components_indices, y_pred_grid, x_test_normalized,
            grid_shape, value, l_limit, u_limit)


# %% Exporting Data for external plotting
if len(components_info_plot[0]) > 0:
    num_points_plot = [[] for _ in range(len(components_info_plot[0]))] 
    mean_pce_plot = [[] for _ in range(len(components_info_plot[0]))]
    fillfactor_plot = [[] for _ in range(len(components_info_plot[0]))]
    stats_p16_plot = [[] for _ in range(len(components_info_plot[0]))]
    stats_p84_plot = [[] for _ in range(len(components_info_plot[0]))]
    remaining_components_plot = []

    for index1, components_info in enumerate(components_info_plot):
        counter = 0 
        for index, component in enumerate(components_info):
            num_points_plot[index].append(component['num_points'])
            mean_pce_plot[index].append(component['mean_pce'])
            fillfactor_plot[index].append(component['fill_factor'])
            stats_p16_plot[index].append(component['stats_p16'])
            stats_p84_plot[index].append(component['stats_p84'])
            
            if num_points_plot[index][index1] > 1: 
                counter += 1
        
        remaining_components_plot.append(counter)

    # %%
    # Define a dictionary to store each array with corresponding filename
    plot_data = {
        "components_num_points_plot.csv": num_points_plot,
        "components_mean_pce_plot.csv": mean_pce_plot,
        "components_fillfactor_plot.csv": fillfactor_plot,
        "components_stats_p16_plot.csv": stats_p16_plot,
        "components_stats_p84_plot.csv": stats_p84_plot
    }

    # Save all datasets to CSV files
    for filename, data in plot_data.items():
        save_to_csv(filename, data, persistence_thresholds, directory_data)

    # %%
    # Export the remaining components data
    export_df = pd.DataFrame({
        'Thresholds': persistence_thresholds,
        'Remaining Components': remaining_components_plot
    })
    
    output_file = os.path.join(directory_data, 'components_remaining.csv')
    export_df.to_csv(output_file, index=False)
    print(f"CSV file '{output_file}' has been created successfully!")
else:
    print("No components found to analyze across thresholds.")