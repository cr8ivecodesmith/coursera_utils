import os
import re
import argparse
from collections import defaultdict


def combine_files_by_module(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Regex to match the filename pattern
    pattern = re.compile(r"(m\d+)-\d+ - (.+)\.txt")

    # Dictionary to group contents by module
    grouped_files = defaultdict(list)

    # Read files and group by module
    for filename in os.listdir(input_directory):
        match = pattern.match(filename)
        if match:
            module, title = match.groups()
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                grouped_files[module].append((title, content))

    # Write combined content to new files
    for module, sections in grouped_files.items():
        output_filepath = os.path.join(
            output_directory, f"{module}_combined.txt")
        with open(output_filepath, 'w', encoding='utf-8') as output_file:
            for title, content in sections:
                output_file.write(f"## {title}\n")
                output_file.write(content)
                output_file.write("\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine files by module based on their naming pattern.")
    parser.add_argument(
        "input_directory",
        help="Path to the directory containing input text files.")
    parser.add_argument(
        "output_directory",
        help="Path to the directory to save combined output files.")
    args = parser.parse_args()

    combine_files_by_module(args.input_directory, args.output_directory)


if __name__ == "__main__":
    main()
