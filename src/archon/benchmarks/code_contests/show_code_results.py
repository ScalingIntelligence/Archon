import os
import yaml
import argparse


def count_correct_yaml_files(eval_path):
    """
    Counts the number of YAML files in the specified directory
    where the key "is_corrects" is set to True in the first element of the list.

    Parameters:
        directory_path (str): The path to the directory containing YAML files.

    Returns:
        total_files (int): The total number of YAML files processed.
        correct_files_count (int): The number of files with "is_corrects" set to True.
    """
    file_names = []
    correct_files_count = 0

    # Iterate through the files in the directory
    for file_name in os.listdir(eval_path):
        file_path = os.path.join(eval_path, file_name)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Append the file name to the list
            file_names.append(file_name)

            # Attempt to read and parse the YAML file
            with open(file_path, "r") as file:
                try:
                    content = yaml.safe_load(file)

                    # Check if "is_corrects" exists and is a list
                    if content.get("is_corrects") and isinstance(
                        content["is_corrects"], list
                    ):
                        # If the first element is True, increment the count
                        if any(content["is_corrects"]) == True:
                            correct_files_count += 1
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML file {file_name}: {e}")

    # Return the total files and correct count
    return len(file_names), correct_files_count


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_path",
        type=str,
        help="The path to the directory containing evaluation files.",
    )
    args = parser.parse_args()

    # Count the correct YAML files
    total_files, correct_files_count = count_correct_yaml_files(args.eval_path)

    # Output the results
    print("=" * 40)
    print(f"Directory: {args.eval_path}")
    print(f"Total eval files: {total_files}")
    print(f"Files with 'is_corrects' set to True: {correct_files_count}")
    print("=" * 40)


if __name__ == "__main__":
    main()
