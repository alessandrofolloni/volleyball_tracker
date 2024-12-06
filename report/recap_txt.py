import re
import pandas as pd
from pathlib import Path

# Base directory (modify this path if necessary)
base_dir = Path('volleyball_tracker_training')  # Modify this path if necessary

# Verify that the base directory exists
if not base_dir.exists():
    print(f"The base directory '{base_dir}' does not exist. Check the path.")
    exit()

# Pattern to find files starting with 'metrics_report_' and ending with '.txt'
report_prefix = 'metrics_report_'
report_suffix = '.txt'

# List to collect data
data = []

# Set to collect all metric names
all_metrics = set()

def parse_model_name(model_name):
    """
    Parses the model name string to extract configuration parameters.

    Args:
        model_name (str): The model name string.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    params = {}
    parts = model_name.split('_')
    for part in parts:
        if part.startswith('epochs'):
            params['Epochs'] = int(part[len('epochs'):])
        elif part.startswith('model'):
            params['Model'] = part[len('model'):]
        elif part.startswith('bs'):
            params['Batch Size'] = int(part[len('bs'):])
        elif part.startswith('opt'):
            params['Optimizer'] = part[len('opt'):]
        elif part.startswith('lr'):
            params['Learning Rate'] = float(part[len('lr'):])
        elif part.startswith('momentum'):
            params['Momentum'] = float(part[len('momentum'):])
        elif part.startswith('wd'):
            params['Weight Decay'] = float(part[len('wd'):])
        else:
            # Assume the first part is 'Subset Name'
            if 'Subset Name' not in params:
                params['Subset Name'] = part
    return params

# Function to extract metrics from a report
def extract_metrics(file_path):
    metrics = {}
    try:
        with file_path.open('r', encoding='utf-8') as file:
            for line in file:
                # Remove any leading/trailing whitespace
                line = line.strip()

                # Extract 'Model Name'
                if line.startswith('- Model name:'):
                    model_match = re.search(r'- Model name:\s*(.+)', line)
                    if model_match:
                        model_name = model_match.group(1).strip()
                        metrics['Model Name'] = model_name
                        # Parse the model name to get configuration parameters
                        config_params = parse_model_name(model_name)
                        metrics.update(config_params)

                # Extract 'Precision'
                elif line.startswith('- Precision (P):'):
                    precision_match = re.search(r'- Precision \(P\):\s*([0-9.]+)', line)
                    if precision_match:
                        metrics['Precision'] = float(precision_match.group(1))

                # Extract 'Recall'
                elif line.startswith('- Recall (R):'):
                    recall_match = re.search(r'- Recall \(R\):\s*([0-9.]+)', line)
                    if recall_match:
                        metrics['Recall'] = float(recall_match.group(1))

                # Extract 'Mean Average Precision @ IoU=0.5 (mAP@0.5):'
                elif line.startswith('- Mean Average Precision @ IoU=0.5 (mAP@0.5):'):
                    map_05_match = re.search(r'- Mean Average Precision @ IoU=0\.5 \(mAP@0\.5\):\s*([0-9.]+)', line)
                    if map_05_match:
                        metrics['mAP@0.5'] = float(map_05_match.group(1))

                # Extract 'Mean Average Precision @ IoU=0.5:0.95 (mAP@0.5:0.95):'
                elif line.startswith('- Mean Average Precision @ IoU=0.5:0.95 (mAP@0.5:0.95):'):
                    map_095_match = re.search(r'- Mean Average Precision @ IoU=0\.5:0\.95 \(mAP@0\.5:0\.95\):\s*([0-9.]+)', line)
                    if map_095_match:
                        metrics['mAP@0.5:0.95'] = float(map_095_match.group(1))

                # Extract 'Average Training Loss'
                elif line.startswith('- Average Training Loss:'):
                    train_loss_match = re.search(r'- Average Training Loss:\s*([0-9.]+)', line)
                    if train_loss_match:
                        metrics['Average Training Loss'] = float(train_loss_match.group(1))

                # Extract 'Average Validation Loss'
                elif line.startswith('- Average Validation Loss:'):
                    val_loss_match = re.search(r'- Average Validation Loss:\s*([0-9.]+)', line)
                    if val_loss_match:
                        metrics['Average Validation Loss'] = float(val_loss_match.group(1))

                # You can add more metrics extraction here if needed

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    return metrics if metrics else None

# Traverse through all subdirectories and collect reports
print(f"Starting search for files in '{base_dir}'...")
for file_path in base_dir.rglob(f'{report_prefix}*{report_suffix}'):
    print(f"Found file: {file_path}")
    metrics = extract_metrics(file_path)
    if metrics:
        metrics['Path'] = str(file_path.parent)  # Add the path for reference
        data.append(metrics)
        all_metrics.update(metrics.keys())
        print(f"Metrics extracted for {file_path}: {metrics}")
    else:
        print(f"No metrics extracted from {file_path}. Skipping this file.")

if not data:
    print("No matching files found or missing metrics in the found files.")
    exit()

# Identify all present metrics
all_metrics.discard('Path')  # 'Path' is not a metric

# Create a DataFrame
df = pd.DataFrame(data)

# Reorder columns
# Define the desired order of columns
ordered_columns = ['Model Name', 'Subset Name', 'Epochs', 'Model', 'Batch Size', 'Optimizer', 'Learning Rate', 'Momentum', 'Weight Decay', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'Average Training Loss', 'Average Validation Loss', 'Path']

# Keep only the columns that are present in the DataFrame
ordered_columns = [col for col in ordered_columns if col in df.columns]

df = df.reindex(columns=ordered_columns)

# Identify the best metrics for each numeric column
best_metrics = {}
for metric in df.columns:
    if metric not in ['Model Name', 'Subset Name', 'Model', 'Optimizer', 'Path'] and pd.api.types.is_numeric_dtype(df[metric]):
        if metric in ['Average Training Loss', 'Average Validation Loss']:
            # For loss metrics, lower is better
            best_metrics[metric] = df[metric].min()
        else:
            # For other metrics, higher is better
            best_metrics[metric] = df[metric].max()

print(f"Identified best metrics: {best_metrics}")

# Function to format metrics, highlighting the best ones
def format_metric(value, metric_name):
    if pd.isna(value):
        return 'N/A'
    if metric_name in best_metrics:
        if metric_name in ['Average Training Loss', 'Average Validation Loss']:
            if value == best_metrics[metric_name]:
                return f"**{value}**"
        else:
            if value == best_metrics[metric_name]:
                return f"**{value}**"
    return f"{value}"

# Create the summary file in Markdown
summary_file = Path('summary_metrics.md')

with summary_file.open('w', encoding='utf-8') as f:
    # Write the header
    header = '| ' + ' | '.join(ordered_columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(ordered_columns)) + ' |'
    f.write(header + '\n')
    f.write(separator + '\n')

    # Write each row with formatted metrics
    for _, row in df.iterrows():
        row_data = []
        for col in ordered_columns:
            if col in ['Model Name', 'Subset Name', 'Model', 'Optimizer', 'Path']:
                row_data.append(str(row[col]))
            else:
                row_data.append(format_metric(row[col], col))
        f.write('| ' + ' | '.join(row_data) + ' |\n')

print(f"Summary file created: {summary_file.resolve()}")