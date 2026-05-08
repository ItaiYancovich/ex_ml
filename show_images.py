import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

folder = "Train Set (Labeled)"
all_images = glob.glob(os.path.join(folder, "*.pgm"))

# Pick one image per person (group by person ID, e.g. p0, p1, ...)
people = {}
for path in all_images:
    name = os.path.basename(path)
    match = re.match(r"(p\d+)_i", name)
    if match:
        person_id = match.group(1)
        if person_id not in people:
            people[person_id] = path

# Take 10 different people
images = [people[k] for k in sorted(people, key=lambda x: int(x[1:]))[:10]]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, img_path in zip(axes.flat, images):
    img = mpimg.imread(img_path)
    ax.imshow(img, cmap="gray")
    person = re.match(r"(p\d+)_i", os.path.basename(img_path)).group(1)
    ax.set_title(person, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()
