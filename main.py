# main.py

import random
import pandas as pd
import pygame
import csv
import time
from screeninfo import get_monitors
import hengam  # Import the hengam package

# Initialize pygame
pygame.init()

# Constants
first_monitor = get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = first_monitor.width, first_monitor.height
FONT_SIZE = 50
FONT_COLOR = (0, 0, 0)
BG_COLOR = (255, 255, 255)
MARGIN = 30
LINE_SPACING = 25
UPWARD_MARGIN = 12  # Adjust this value as needed
DOWNWARD_MARGIN = 12  # Adjust this value as needed

# Setup screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
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

        # Adjust the bounding box upwards and downwards
        word_rect.top -= UPWARD_MARGIN
        word_rect.height += UPWARD_MARGIN + DOWNWARD_MARGIN

        # Draw word
        screen.blit(word_surface, (word_rect.left, word_rect.top + UPWARD_MARGIN))

        # Record bounding box
        word_boxes.append((word, word_rect))

        # Move to the right for the next word
        current_x = word_rect.right + 5  # Add small spacing between words

    return word_boxes

# Load CSV data
def load_questions(csv_file):
    questions = {}
    with open(csv_file, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter='\t')
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

# Main loop to display questions
def display_questions(questions):
    bounding_boxes = []
    results = []
    gaze_data_all = []  # To store all gaze data

    question_numbers = list(questions.keys())
    random.shuffle(question_numbers)

    for qnr in question_numbers:
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
                min(y_coords) - UPWARD_MARGIN,
                max(x_coords),
                max(y_coords) + DOWNWARD_MARGIN
            )
            text_regions.append(text_region)

        # Initialize the eye tracker with your custom package
        LICENSE_KEY = "int.lab2024"
        try:
            tracker = hengam.EyeTracker(SCREEN_WIDTH, SCREEN_HEIGHT, 0.0, LICENSE_KEY)
        except RuntimeError as e:
            print(f"Failed to initialize EyeTracker: {e}")
            pygame.quit()
            return bounding_boxes, results, gaze_data_all

        if not tracker.isLicenseValid():
            print("License is invalid or expired. Exiting...")
            pygame.quit()
            return bounding_boxes, results, gaze_data_all

        # Clear the data buffer before starting recording
        tracker.clearDataBuffer()

        # Start eye-tracking
        try:
            tracker.start()
        except RuntimeError as e:
            print(f"Error starting eye tracker: {e}")
            pygame.quit()
            return bounding_boxes, results, gaze_data_all

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
            # Update eye-tracking data
            try:
                tracker.update()
            except RuntimeError as e:
                print(f"Error during recording: {e}")
                tracker.stop()
                pygame.quit()
                return bounding_boxes, results, gaze_data_all

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

        # Stop eye-tracking
        tracker.stop()

        # Fetch gaze data for this question
        try:
            results_data = tracker.getFormattedData()
            for data_point in results_data:
                data_point['qnr'] = qnr  # Associate gaze data with question number
            gaze_data_all.extend(results_data)
        except RuntimeError as e:
            print(f"Error fetching data: {e}")
            pygame.quit()
            return bounding_boxes, results, gaze_data_all

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
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
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
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
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
    csv_file = "realSurveyQuestions.csv"  # Input CSV file
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