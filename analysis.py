import pandas as pd
import numpy as np
from shapely.geometry import Point, box
from rtree import index
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from screeninfo import get_monitors
import seaborn as sns

# Constants
first_monitor = get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = first_monitor.width, first_monitor.height

# Load gaze data
gaze_data = pd.read_csv('gaze_data.csv')

# Flip the y-coordinate to match the plotting coordinate system
gaze_data['y'] = SCREEN_HEIGHT - gaze_data['y']

# Filter valid gaze data (validity == 'Valid')
valid_gaze_data = gaze_data[gaze_data['validity'] == 'Valid'].copy()

# Ensure 'qnr' is integer type
valid_gaze_data.loc[:, 'qnr'] = valid_gaze_data['qnr'].astype(int)

print(valid_gaze_data[['x', 'y']].describe())

# Load bounding boxes
bounding_boxes = pd.read_csv('bounding_boxes.csv')

# Parse bounding box coordinates
def parse_tuple(s):
    return ast.literal_eval(s)

bounding_boxes['top_left'] = bounding_boxes['top_left'].apply(parse_tuple)
bounding_boxes['bottom_right'] = bounding_boxes['bottom_right'].apply(parse_tuple)

# Expand bounding box coordinates
bounding_boxes['x_min'] = bounding_boxes['top_left'].apply(lambda t: t[0])
bounding_boxes['y_min'] = bounding_boxes['top_left'].apply(lambda t: t[1])
bounding_boxes['x_max'] = bounding_boxes['bottom_right'].apply(lambda t: t[0])
bounding_boxes['y_max'] = bounding_boxes['bottom_right'].apply(lambda t: t[1])

# Flip the y-coordinates of bounding boxes
bounding_boxes['y_min'] = SCREEN_HEIGHT - bounding_boxes['y_min']
bounding_boxes['y_max'] = SCREEN_HEIGHT - bounding_boxes['y_max']

# Swap y_min and y_max if necessary
bounding_boxes['y_min'], bounding_boxes['y_max'] = (
    np.minimum(bounding_boxes['y_min'], bounding_boxes['y_max']),
    np.maximum(bounding_boxes['y_min'], bounding_boxes['y_max'])
)

# Ensure 'qnr' is integer type
bounding_boxes.loc[:, 'qnr'] = bounding_boxes['qnr'].astype(int)

print(bounding_boxes[['x_min', 'y_min', 'x_max', 'y_max']].describe())
print(bounding_boxes[['top_left', 'bottom_right']].head())

# Function to match gaze points to words
def match_gaze_to_words(gaze_data, bounding_boxes):
    matched_data = []
    grouped_gaze = gaze_data.groupby('qnr')
    grouped_boxes = bounding_boxes.groupby('qnr')

    for qnr, gaze_group in grouped_gaze:
        if qnr not in grouped_boxes.groups:
            print(f"No bounding boxes for question {qnr}")
            continue

        boxes_qnr = bounding_boxes[bounding_boxes['qnr'] == qnr].reset_index()
        idx = index.Index()
        for i, row in boxes_qnr.iterrows():
            idx.insert(i, (row['x_min'], row['y_min'], row['x_max'], row['y_max']), obj=row)

        for _, gaze_point in gaze_group.iterrows():
            point = Point(gaze_point['x'], gaze_point['y'])
            possible_matches = list(idx.intersection((gaze_point['x'], gaze_point['y'], gaze_point['x'], gaze_point['y']), objects=True))

            if possible_matches:
                for item in possible_matches:
                    row = item.object
                    word_box = box(row['x_min'], row['y_min'], row['x_max'], row['y_max'])
                    if word_box.contains(point):
                        matched_data.append({
                            'qnr': qnr,
                            'timestamp': gaze_point['timestamp_us'],
                            'x': gaze_point['x'],
                            'y': gaze_point['y'],
                            'word': row.get('word', None),
                            'label': row.get('label', None),
                            'ai': row.get('ai', None),
                            'topic': row.get('topic', None)
                        })
                        break
            else:
                matched_data.append({
                    'qnr': qnr,
                    'timestamp': gaze_point['timestamp_us'],
                    'x': gaze_point['x'],
                    'y': gaze_point['y'],
                    'word': None,
                    'label': None,
                    'ai': None,
                    'topic': None
                })

    return pd.DataFrame(matched_data)

# Perform the matching
matched_gaze_data = match_gaze_to_words(valid_gaze_data, bounding_boxes)

# Save matched gaze data to a CSV (optional)
matched_gaze_data.to_csv('matched_gaze_data.csv', index=False)

# Filter out unmatched gaze points
if 'word' in matched_gaze_data.columns:
    matched_gaze_points = matched_gaze_data.dropna(subset=['word'])
else:
    matched_gaze_points = pd.DataFrame(columns=matched_gaze_data.columns)

# Count the number of gaze points per word
word_focus_counts = matched_gaze_points.groupby(['qnr', 'label', 'word']).size().reset_index(name='gaze_count')

# Display the top focused words
top_words = word_focus_counts.sort_values(by='gaze_count', ascending=False)
print(top_words.head(10))

# Plotting for a specific question and text
def plot_top_words(word_counts, qnr, label, top_n=5):
    word_counts_qnr_label = word_counts[(word_counts['qnr'] == qnr) & (word_counts['label'] == label)]
    top_words = word_counts_qnr_label.sort_values(by='gaze_count', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(top_words['word'], top_words['gaze_count'], color='skyblue')
    plt.title(f'Top {top_n} Focused Words in {label} (Question {qnr})')
    plt.xlabel('Word')
    plt.ylabel('Number of Gaze Points')
    plt.legend([f'Gaze counts for words in {label}'])
    plt.tight_layout()
    plt.show()

# Example usage
qnr = 1  # Question number
plot_top_words(word_focus_counts, qnr, 'Text A')
plot_top_words(word_focus_counts, qnr, 'Text B')

def plot_gaze_and_boxes(qnr):
    # Filter data for the question
    gaze_qnr = valid_gaze_data[valid_gaze_data['qnr'] == qnr]
    boxes_qnr = bounding_boxes[bounding_boxes['qnr'] == qnr]

    # Set up the figure size based on screen resolution and DPI
    dpi = 100  # Adjust DPI as needed
    fig_width = SCREEN_WIDTH / dpi
    fig_height = SCREEN_HEIGHT / dpi
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    ax = plt.gca()
    plt.scatter(gaze_qnr['x'], gaze_qnr['y'], color='blue', s=5, label='Gaze Points')

    # Add bounding boxes
    for _, row in boxes_qnr.iterrows():
        rect = Rectangle(
            (row['x_min'], row['y_min']),
            row['x_max'] - row['x_min'],
            row['y_max'] - row['y_min'],
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    # Set aspect ratio to equal to prevent stretching
    ax.set_aspect('equal', adjustable='box')

    # Set limits to match screen dimensions
    plt.xlim(0, SCREEN_WIDTH)
    plt.ylim(0, SCREEN_HEIGHT)

    # Invert y-axis
    plt.gca().invert_yaxis()

    plt.title(f'Gaze Points and Bounding Boxes for Question {qnr}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
plot_gaze_and_boxes(qnr)

def plot_raw_gaze(qnr):
    # Filter data for the question
    gaze_qnr = valid_gaze_data[valid_gaze_data['qnr'] == qnr]

    plt.figure(figsize=(12, 8))
    plt.scatter(gaze_qnr['x'], gaze_qnr['y'], s=10, c='red', alpha=0.5, label='Raw Gaze Points')
    plt.title(f'Raw Gaze Points for Question {qnr}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Plot raw gaze data
plot_raw_gaze(qnr)

def plot_gaze_heatmap(qnr):
    # Filter data for the question
    gaze_qnr = valid_gaze_data[valid_gaze_data['qnr'] == qnr]

    plt.figure(figsize=(12, 8))
    sns.kdeplot(
        x=gaze_qnr['x'], y=gaze_qnr['y'],
        cmap='viridis', fill=True, alpha=0.7,
        bw_adjust=0.5, levels=100, thresh=0
    )
    plt.title(f'Gaze Heatmap for Question {qnr}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Plot gaze heatmap
plot_gaze_heatmap(qnr)

def detect_fixations_ivt(data, velocity_threshold):
    fixations = []
    current_fixation = []

    data = data.sort_values('timestamp_us').reset_index(drop=True)

    for i in range(len(data) - 1):
        dx = data.iloc[i + 1]['x'] - data.iloc[i]['x']
        dy = data.iloc[i + 1]['y'] - data.iloc[i]['y']
        dt = data.iloc[i + 1]['timestamp_us'] - data.iloc[i]['timestamp_us']

        if dt == 0:
            velocity = float('inf')
        else:
            velocity = np.sqrt(dx**2 + dy**2) / dt

        if velocity <= velocity_threshold:
            current_fixation.append(data.iloc[i])
        else:
            if current_fixation:
                fixation_df = pd.DataFrame(current_fixation)
                centroid_x = fixation_df['x'].mean()
                centroid_y = fixation_df['y'].mean()
                start_time = fixation_df['timestamp_us'].min()
                end_time = fixation_df['timestamp_us'].max()
                fixations.append((start_time, end_time, centroid_x, centroid_y))
                current_fixation = []

    if current_fixation:
        fixation_df = pd.DataFrame(current_fixation)
        centroid_x = fixation_df['x'].mean()
        centroid_y = fixation_df['y'].mean()
        start_time = fixation_df['timestamp_us'].min()
        end_time = fixation_df['timestamp_us'].max()
        fixations.append((start_time, end_time, centroid_x, centroid_y))

    return fixations

def plot_fixations(qnr, velocity_threshold=50):
    gaze_qnr = valid_gaze_data[valid_gaze_data['qnr'] == qnr]

    fixations_ivt = detect_fixations_ivt(gaze_qnr, velocity_threshold)
    fixation_df_ivt = pd.DataFrame(fixations_ivt, columns=['start_time', 'end_time', 'centroid_x', 'centroid_y'])

    plt.figure(figsize=(12, 8))
    plt.scatter(
        fixation_df_ivt['centroid_x'], fixation_df_ivt['centroid_y'],
        c='blue', s=100, label='Fixations'
    )

    # Annotate fixation numbers
    for i, row in fixation_df_ivt.iterrows():
        plt.text(row['centroid_x'], row['centroid_y'], str(i + 1), color='white', fontsize=12)

    plt.title(f'Fixations Detected by I-VT for Question {qnr}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Plot fixations
plot_fixations(qnr)