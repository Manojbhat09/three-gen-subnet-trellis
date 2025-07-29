import re

def calculate_reliability(log_file_path):
    """
    Parses a log file to calculate the task reliability.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        None: Prints the results to the console.
    """
    total_tasks = 0
    failed_tasks = 0

    # Regular expression to find lines with "Task fidelity"
    fidelity_pattern = re.compile(r"Task fidelity: (\d\.\d+)")

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = fidelity_pattern.search(line)
                if match:
                    total_tasks += 1
                    fidelity_score = float(match.group(1))
                    if fidelity_score == 0.0:
                        failed_tasks += 1

        if total_tasks > 0:
            successful_tasks = total_tasks - failed_tasks
            reliability = (successful_tasks / total_tasks) * 100
            print(f"--- Analysis of {log_file_path} ---")
            print(f"Total Validated Tasks: {total_tasks}")
            print(f"Successful Tasks (fidelity > 0.0): {successful_tasks}")
            print(f"Failed Tasks (fidelity == 0.0): {failed_tasks}")
            print(f"Reliability: {reliability:.2f}%")
        else:
            print(f"No tasks with fidelity scores found in {log_file_path}.")

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- To use this script ---
# 1. Make sure the log file 'continuous_trellis5.log' is in the same directory
#    as this script.
# 2. Run the script.

if __name__ == '__main__':
    # Since I am running this in my environment, I will use the path
    # where you uploaded the file.
    calculate_reliability('continuous_trellis.log.old1')
