import random
import pandas as pd
import pygame
import csv
import time
from screeninfo import get_monitors

# Initialize pygame
pygame.init()

# Constants
first_monitor = get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = first_monitor.width, first_monitor.height
FONT_SIZE = 32
FONT_COLOR = (0, 0, 0)
BG_COLOR = (255, 255, 255)
MARGIN = 50
LINE_SPACING = 10

# Setup screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Text Comparison Study")
font = pygame.font.Font(None, FONT_SIZE)

# Function to render text and calculate bounding boxes
def render_text(screen, text, x, y, font, max_width):
    words = text.split()
    word_boxes = []
    current_x, current_y = x, y

    for word in words:
        word_surface = font.render(word, True, FONT_COLOR)
        word_rect = word_surface.get_rect()
        word_rect.topleft = (current_x, current_y)

        # If word exceeds max width, move to next line
        if word_rect.right > x + max_width:
            current_x = x
            current_y += FONT_SIZE + LINE_SPACING
            word_rect.topleft = (current_x, current_y)

        # Draw word
        screen.blit(word_surface, word_rect.topleft)

        # Record bounding box
        word_boxes.append((word, word_rect))

        # Move to the right for the next word
        current_x = word_rect.right + 5  # Add small spacing between words

    return word_boxes

# Load CSV data
def load_questions(csv_file):
    questions = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            qnr = int(row["qnr"])
            if qnr not in questions:
                questions[qnr] = []
            questions[qnr].append({
                "text": row["text"],
                "ai": row["ai"].strip().lower() == 'true',
                "topic": row["topic"]
            })
    return questions

# Mock EyeTracker class
class MockEyeTracker:
    def __init__(self, screen_width, screen_height, license_key, text_regions):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.license_key = license_key
        self.is_recording = False
        self.data_buffer = []
        self.start_time = None
        self.text_regions = text_regions  # List of text regions

    def isLicenseValid(self):
        # Simulate license validation
        return True

    def clearDataBuffer(self):
        self.data_buffer = []

    def start(self):
        self.is_recording = True
        self.start_time = time.time()

    def pause(self):
        self.is_recording = False

    def stop(self):
        self.is_recording = False

    def update(self):
        if self.is_recording and self.text_regions:
            # Simulate gaze data within text regions
            timestamp = time.time() - self.start_time

            # Randomly select a text region (simulate reading)
            region = random.choice(self.text_regions)
            x_min, y_min, x_max, y_max = region

            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)

            validity = random.choice([0, 1])  # 0 for valid, 1 for invalid
            self.data_buffer.append({
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'validity': validity
            })
            # Simulate sampling rate (e.g., 60 Hz)
            time.sleep(1 / 60)

    def getFormattedData(self):
        # Return the data buffer
        return self.data_buffer

# Main loop to display questions
def display_questions(questions):
    bounding_boxes = []
    results = []
    gaze_data_all = []  # To store all gaze data

    for qnr in sorted(questions.keys()):
        texts = questions[qnr]

        # Randomize the order of texts
        random.shuffle(texts)

        # Assign labels
        labels = ["Text A", "Text B"]

        # Display both texts
        screen.fill(BG_COLOR)
        y_position = MARGIN

        # Calculate width for each text
        text_width = (SCREEN_WIDTH - 3 * MARGIN) // 2

        word_boxes = {}
        text_regions = []  # To store text regions for gaze simulation

        # Draw labels and texts
        for idx, text_info in enumerate(texts):
            x_position = MARGIN + idx * (text_width + MARGIN)

            # Draw label
            label_surface = font.render(labels[idx], True, FONT_COLOR)
            screen.blit(label_surface, (x_position, y_position))

            # Render text
            boxes = render_text(
                screen,
                text_info["text"],
                x_position,
                y_position + FONT_SIZE + LINE_SPACING,
                font,
                text_width
            )

            # Collect word boxes
            word_boxes[labels[idx]] = {
                "text_info": text_info,
                "boxes": [(word, rect.topleft, rect.bottomright) for word, rect in boxes]
            }

            # Calculate the bounding box of the entire text
            x_coords = [rect.left for _, rect in boxes] + [rect.right for _, rect in boxes]
            y_coords = [rect.top for _, rect in boxes] + [rect.bottom for _, rect in boxes]
            text_region = (
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords)
            )
            text_regions.append(text_region)

        # Initialize the eye tracker with text regions
        tracker = MockEyeTracker(SCREEN_WIDTH, SCREEN_HEIGHT, "dummy_license_key", text_regions)

        if not tracker.isLicenseValid():
            print("License is invalid or expired. Exiting...")
            pygame.quit()
            return bounding_boxes, results, gaze_data_all

        # Clear the data buffer before starting recording
        tracker.clearDataBuffer()
        tracker.start()  # Start eye-tracking

        # Create selection buttons
        button_font = pygame.font.Font(None, 32)
        buttons = []
        button_width = 200
        button_height = 50
        button_y = SCREEN_HEIGHT - MARGIN - button_height

        for idx in range(2):
            button_x = MARGIN + idx * (SCREEN_WIDTH - 2 * MARGIN - button_width)
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            buttons.append((button_rect, labels[idx]))

        # Draw buttons
        for button_rect, label in buttons:
            pygame.draw.rect(screen, (200, 200, 200), button_rect)
            button_label = button_font.render(f"Select {label} as AI", True, FONT_COLOR)
            label_rect = button_label.get_rect(center=button_rect.center)
            screen.blit(button_label, label_rect)

        # Update the display after drawing texts and buttons
        pygame.display.flip()

        # Start timing after everything is displayed
        start_time = time.time()

        # Wait for participant to make a selection
        selection_made = False
        selected_text = None

        while not selection_made:
            tracker.update()  # Update eye-tracking data
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    tracker.stop()
                    pygame.quit()
                    return bounding_boxes, results, gaze_data_all
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    for button_rect, label in buttons:
                        if button_rect.collidepoint(mouse_pos):
                            selected_text = label
                            selection_made = True
                            break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        tracker.stop()
                        pygame.quit()
                        return bounding_boxes, results, gaze_data_all

        # Participant made a selection; end timing
        end_time = time.time()
        total_time_spent = end_time - start_time
        tracker.stop()  # Stop eye-tracking

        # Fetch gaze data for this question
        gaze_data = tracker.getFormattedData()
        for data_point in gaze_data:
            data_point['qnr'] = qnr  # Associate gaze data with question number
        gaze_data_all.extend(gaze_data)

        # Record results
        selected_ai = selected_text
        actual_ai = labels[texts.index(next(t for t in texts if t["ai"]))]

        results.append({
            "qnr": qnr,
            "selected": selected_ai,
            "actual_ai": actual_ai,
            "correct": selected_ai == actual_ai,
            "total_time_spent": total_time_spent
        })

        # Save bounding boxes
        for label in labels:
            data = {
                "qnr": qnr,
                "label": label,
                "ai": word_boxes[label]["text_info"]["ai"],
                "topic": word_boxes[label]["text_info"]["topic"],
                "word_boxes": word_boxes[label]["boxes"]
            }
            bounding_boxes.append(data)

    return bounding_boxes, results, gaze_data_all

# Save bounding boxes to a CSV
def save_bounding_boxes(bounding_boxes, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["qnr", "label", "ai", "topic", "word", "top_left", "bottom_right"])
        for data in bounding_boxes:
            for word, top_left, bottom_right in data["word_boxes"]:
                writer.writerow([
                    data["qnr"],
                    data["label"],
                    data["ai"],
                    data["topic"],
                    word,
                    top_left,
                    bottom_right
                ])

# Save results to a CSV
def save_results(results, output_file):
    with open(output_file, 'w', newline='') as file:
        fieldnames = ["qnr", "selected", "actual_ai", "correct", "total_time_spent"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "qnr": result["qnr"],
                "selected": result["selected"],
                "actual_ai": result["actual_ai"],
                "correct": result["correct"],
                "total_time_spent": result["total_time_spent"]
            })

# Save gaze data to a CSV
def save_gaze_data(gaze_data_all, output_file):
    df = pd.DataFrame(gaze_data_all)
    df.to_csv(output_file, index=False)

# Main function
def main():
    csv_file = "questions.csv"  # Input CSV file
    bounding_boxes_file = "bounding_boxes.csv"  # Output CSV file for bounding boxes
    results_file = "results.csv"  # Output CSV file for participant selections
    gaze_data_file = "gaze_data.csv"  # Output CSV file for gaze data

    questions = load_questions(csv_file)
    bounding_boxes, results, gaze_data_all = display_questions(questions)
    save_bounding_boxes(bounding_boxes, bounding_boxes_file)
    save_results(results, results_file)
    save_gaze_data(gaze_data_all, gaze_data_file)
    print("Bounding boxes saved to:", bounding_boxes_file)
    print("Results saved to:", results_file)
    print("Gaze data saved to:", gaze_data_file)

if __name__ == "__main__":
    main()
