import pandas as pd
from shapely.geometry import Point, box
from rtree import index
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from screeninfo import get_monitors

# Can also be set manually if resolution should be different.
first_monitor = get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = first_monitor.width, first_monitor.height

# Load gaze data
gaze_data = pd.read_csv('gaze_data.csv')
print(gaze_data[['x', 'y']].describe())

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

print(bounding_boxes[['x_min', 'y_min', 'x_max', 'y_max']].describe())
print(bounding_boxes[['top_left', 'bottom_right']].head())

# Filter valid gaze data
valid_gaze_data = gaze_data[gaze_data['validity'] == 0]

# Function to match gaze points to words
def match_gaze_to_words(gaze_data, bounding_boxes):
    matched_data = []

    # Group gaze data and bounding boxes by question number
    grouped_gaze = gaze_data.groupby('qnr')
    grouped_boxes = bounding_boxes.groupby('qnr')

    for qnr, gaze_group in grouped_gaze:
        if qnr not in grouped_boxes.groups:
            continue

        # Get bounding boxes for this question
        boxes_qnr = bounding_boxes[bounding_boxes['qnr'] == qnr].reset_index()

        # Create R-tree index of the bounding boxes
        idx = index.Index()
        for i, row in boxes_qnr.iterrows():
            idx.insert(i, (row['x_min'], row['y_min'], row['x_max'], row['y_max']))

        for _, gaze_point in gaze_group.iterrows():
            point = Point(gaze_point['x'], gaze_point['y'])
            # Find possible matches
            possible_matches = list(idx.intersection((gaze_point['x'], gaze_point['y'], gaze_point['x'], gaze_point['y'])))
            matched = False
            for i in possible_matches:
                row = boxes_qnr.iloc[i]
                word_box = box(row['x_min'], row['y_min'], row['x_max'], row['y_max'])
                if word_box.contains(point):
                    matched_data.append({
                        'qnr': qnr,
                        'timestamp': gaze_point['timestamp'],
                        'x': gaze_point['x'],
                        'y': gaze_point['y'],
                        'word': row['word'],
                        'label': row['label'],  # Text A or Text B
                        'ai': row['ai'],
                        'topic': row['topic']
                    })
                    matched = True
                    break  # Stop after the first match
            if not matched:
                # Gaze point did not match any word
                matched_data.append({
                    'qnr': qnr,
                    'timestamp': gaze_point['timestamp'],
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
matched_gaze_points = matched_gaze_data.dropna(subset=['word'])

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
    plt.bar(top_words['word'], top_words['gaze_count'])
    plt.title(f'Top {top_n} Focused Words in {label} (Question {qnr})')
    plt.xlabel('Word')
    plt.ylabel('Number of Gaze Points')
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

    # Invert y-axis if your coordinate system has origin at top-left
    plt.gca().invert_yaxis()

    plt.title(f'Gaze Points and Bounding Boxes for Question {qnr}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
plot_gaze_and_boxes(1)
