import matplotlib.pyplot as plt
import numpy as np

def plot_section(file_path, section, hid_value):
    """
    Plot bar charts from judge results file.
    
    Args:
        file_path: Path to judge_results.txt
        section: Section identifier (e.g., "cddd00")
        hid_value: Hidden agenda strength value to filter
    """
    data = {}
    current_section = None

    # Read file
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Detect section header
            if not line.startswith("pub"):
                current_section = line
                data[current_section] = []
            else:
                # Parse pub/hid line
                parts = line.split(":")
                left, right = parts[0].strip(), parts[1].strip()
                pub_num = int(left.split()[1])
                hid_num = int(left.split()[3])
                values = [float(x) for x in right.split()]
                data[current_section].append((pub_num, hid_num, values))

    if section not in data:
        raise ValueError(f"Section {section} not found in file.")

    # Extract only the requested hid
    filtered = [(pub, vals) for pub, hid, vals in data[section] if hid == hid_value]
    filtered.sort(key=lambda x: x[0])  # sort by pub number

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (pub, vals) in enumerate(filtered):
        axes[i].bar(range(1, 4), vals)
        axes[i].set_title(f"public agenda strength {pub}")
        axes[i].set_xticks([1, 2, 3], ["Cat", "Dog", "Undecided"])
        axes[i].set_ylim((0, 11))
    
    plt.tight_layout()
    plt.savefig('src/bars.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to src/bars.png")
    plt.show()

if __name__ == "__main__":
    plot_section("src/data/judge_results.txt", "cddd00", 1)