"""Convert notebook_src.py (# %% cell markers) into palimpsest.ipynb."""
import re, sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

src = open("notebook_src.py").read()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M):
    if not chunk.strip():
        continue
    first, _, body = chunk.partition("\n")
    if "[markdown]" in first:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
        cells.append(new_markdown_cell(md))
    else:
        cells.append(new_code_cell(body.strip("\n")))
nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}})
nbformat.write(nb, "palimpsest.ipynb")
print(f"wrote palimpsest.ipynb with {len(cells)} cells")
