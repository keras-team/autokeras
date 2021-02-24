import base64
import json
import os
import sys
from io import BytesIO

import requests
from PIL import Image


def main(directory):
    contributors = []
    for contributor in json.load(open("contributors.json")):
        if contributor["type"] != "User":
            continue
        if contributor["login"] == "codacy-badger":
            continue
        contributors.append(contributor)

    size = 36
    gap = 3
    elem_per_line = 22
    width = elem_per_line * (size + gap) + gap
    height = ((len(contributors) - 1) // elem_per_line + 1) * (size + gap) + gap

    html = '<svg xmlns="http://www.w3.org/2000/svg" '
    html += 'xmlns:xlink="http://www.w3.org/1999/xlink" '
    html += 'width="{width}" height="{height}">'.format(width=width, height=height)

    defs = "<defs>"
    defs += '<rect id="rect" width="36" height="36" rx="18"/>'
    defs += '<clipPath id="clip"> <use xlink:href="#rect"/> </clipPath> '
    defs += "</defs>"

    html += defs + "\n"

    for index, contributor in enumerate(contributors):
        file_name = os.path.join(directory, str(index) + ".jpeg")
        response = requests.get(contributor["avatar_url"])
        file = open(file_name, "wb")
        file.write(response.content)
        file.close()
        image = Image.open(file_name)
        image = image.resize((size, size))
        # image.convert('RGB').save(file_name)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("UTF-8")

        xi = index % elem_per_line
        yi = index // elem_per_line
        x = xi * (size + gap) + gap
        y = yi * (size + gap) + gap
        temp = (
            '<a xlink:href="{html_url}"> '
            + '<image transform="translate({x},{y})" '
            + 'xlink:href="data:image/png;base64,{img_str}" '
            + 'alt="{login}" clip-path="url(#clip)" '
            + 'width="36" height="36"/></a>'
        )
        temp = temp.format(
            html_url=contributor["html_url"],
            x=x,
            y=y,
            img_str=img_str,
            login=contributor["login"],
        )
        html += temp + "\n"

    html += "</svg>"
    print(html)


if __name__ == "__main__":
    main(sys.argv[1])
