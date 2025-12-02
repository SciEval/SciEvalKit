# Save this code as a file named, for example, calculate_full_accuracy.py

import json
import argparse
import sys
from typing import Dict, List, Any


def check_mcq_correctness(prediction: List[str], ground_truth: Dict[str, float]) -> bool:
    """
    Checks if a multiple-choice question was answered correctly based on exact match.

    Args:
        prediction (List[str]): A list of options selected by the model.
        ground_truth (Dict[str, float]): A dict where keys are options and values
                                        are 1.0 for correct, 0.0 for incorrect.

    Returns:
        bool: True if the selected options exactly match the correct options.
    """
    # Find all correct options from the ground truth
    correct_options = {option for option, value in ground_truth.items() if value == 1.0}

    # Convert the model's prediction to a set for easy comparison
    predicted_options = set(prediction)

    # Check for exact match between the sets
    return correct_options == predicted_options


def calculate_full_accuracy(file_path: str, tolerance: float = 0.01) -> None:
    """
    Loads a JSON result file and calculates MCQ accuracy, numerical accuracy,
    and the overall score (total accuracy).

    Args:
        file_path (str): The path to the input JSON result file.
        tolerance (float): The relative tolerance for numerical answers.
    """
    # --- 1. Load and Validate the JSON file ---
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.", file=sys.stderr)
        sys.exit(1)

    if 'per_question' not in data or not isinstance(data['per_question'], list):
        print(f"Error: JSON file must contain a 'per_question' key with a list of questions.", file=sys.stderr)
        sys.exit(1)

    questions = data['per_question']
    total_questions = len(questions)

    # --- 2. Initialize counters for each question type ---
    mcq_count = 0
    correct_mcq = 0
    numeric_count = 0
    correct_numeric = 0

    # --- 3. Iterate through each question and evaluate ---
    for i, question_data in enumerate(questions):
        q_type = question_data.get('question_type')
        parsed = question_data.get('parsed', {})
        ground_truth = question_data.get('ground_truth')

        if not all([q_type, parsed, ground_truth]):
            print(
                f"Warning: Skipping question {i + 1} due to missing 'question_type', 'parsed', or 'ground_truth' fields.",
                file=sys.stderr)
            continue

        # --- Evaluate MCQ questions ---
        if q_type == 'mcq':
            mcq_count += 1
            selected_options = parsed.get('selected_options', [])
            if isinstance(ground_truth, dict) and isinstance(selected_options, list):
                if check_mcq_correctness(selected_options, ground_truth):
                    correct_mcq += 1
            else:
                print(f"Warning: Skipping MCQ question {i + 1} due to malformed ground_truth or selected_options.",
                      file=sys.stderr)


        # --- Evaluate Numeric questions ---
        elif q_type == 'numeric':
            numeric_count += 1
            predicted_answer = parsed.get('numeric_answer')

            if predicted_answer is None:
                continue  # Model failed to provide a number, so it's incorrect.

            try:
                ground_truth_answer = float(ground_truth)
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not convert numeric ground_truth '{ground_truth}' for question {i + 1}. Skipping.",
                    file=sys.stderr)
                continue

            # Calculate accuracy with tolerance
            allowed_error = tolerance * abs(ground_truth_answer)
            absolute_difference = abs(predicted_answer - ground_truth_answer)

            if absolute_difference <= allowed_error:
                correct_numeric += 1

    # --- 4. Calculate final scores ---
    total_correct = correct_mcq + correct_numeric

    mcq_accuracy = (correct_mcq / mcq_count) * 100 if mcq_count > 0 else 0.0
    numeric_accuracy = (correct_numeric / numeric_count) * 100 if numeric_count > 0 else 0.0
    overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0.0

    # --- 5. Print the comprehensive report ---
    print("\n--- Comprehensive Accuracy Report ---")
    print(f"Total Questions Analyzed: {total_questions}\n")

    print("--- MCQ Performance ---")
    print(f"Total MCQ Questions: {mcq_count}")
    print(f"Correct MCQ Answers: {correct_mcq}")
    print(f"MCQ Accuracy: {mcq_accuracy:.2f}%\n")

    print("--- Numerical Performance ---")
    print(f"Tolerance set to: {tolerance * 100:.2f}%")
    print(f"Total Numerical Questions: {numeric_count}")
    print(f"Correct Numerical Answers: {correct_numeric}")
    print(f"Numerical Accuracy: {numeric_accuracy:.2f}%\n")

    print("--- Overall Score ---")
    print(f"Total Correct Answers (All Types): {total_correct}")
    print(f"Overall Accuracy (Overall Score): {overall_accuracy:.2f}%")
    print("---------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive accuracy (MCQ, Numeric, Overall) from a JSON result file."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON result file."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for numerical correctness. Default is 0.01 (1%)."
    )

    args = parser.parse_args()
    calculate_full_accuracy(args.json_file, args.tolerance)