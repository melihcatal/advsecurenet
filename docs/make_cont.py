import os

docs_path = 'docs/'


def clean_rst_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if 'Subpackages' not in line and 'Submodules' not in line and 'Module contents' not in line:
            new_lines.append(line)

    with open(filepath, 'w') as file:
        file.writelines(new_lines)


def main():
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.rst'):
                clean_rst_file(os.path.join(root, file))


if __name__ == '__main__':
    main()
