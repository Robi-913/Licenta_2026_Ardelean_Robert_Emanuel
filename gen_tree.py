import os

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".conda", ".idea",".gitattributes", ".gitignore"}
SKIP_EXT = {".jpg", ".jpeg"}


def build_tree(folder, prefix, result):
    items = sorted(os.listdir(folder))

    filtered = []
    for name in items:
        ext = os.path.splitext(name)[1].lower()
        if name not in SKIP_DIRS and ext not in SKIP_EXT:
            filtered.append(name)

    for i, name in enumerate(filtered):
        full_path = os.path.join(folder, name)
        last = i == len(filtered) - 1

        branch = "└── " if last else "├── "
        result.append(prefix + branch + name)

        if os.path.isdir(full_path):
            spacer = "    " if last else "│   "
            build_tree(full_path, prefix + spacer, result)


lines = ["."]
build_tree(".", "", lines)

with open("structure.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Saved to structure.txt")