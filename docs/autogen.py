import os
import pathlib
import shutil

import keras_autodoc
import tutobooks

PAGES = {
    "image_classifier.md": [
        "autokeras.ImageClassifier",
        "autokeras.ImageClassifier.fit",
        "autokeras.ImageClassifier.predict",
        "autokeras.ImageClassifier.evaluate",
        "autokeras.ImageClassifier.export_model",
    ],
    "image_regressor.md": [
        "autokeras.ImageRegressor",
        "autokeras.ImageRegressor.fit",
        "autokeras.ImageRegressor.predict",
        "autokeras.ImageRegressor.evaluate",
        "autokeras.ImageRegressor.export_model",
    ],
    "text_classifier.md": [
        "autokeras.TextClassifier",
        "autokeras.TextClassifier.fit",
        "autokeras.TextClassifier.predict",
        "autokeras.TextClassifier.evaluate",
        "autokeras.TextClassifier.export_model",
    ],
    "text_regressor.md": [
        "autokeras.TextRegressor",
        "autokeras.TextRegressor.fit",
        "autokeras.TextRegressor.predict",
        "autokeras.TextRegressor.evaluate",
        "autokeras.TextRegressor.export_model",
    ],
    "structured_data_classifier.md": [
        "autokeras.StructuredDataClassifier",
        "autokeras.StructuredDataClassifier.fit",
        "autokeras.StructuredDataClassifier.predict",
        "autokeras.StructuredDataClassifier.evaluate",
        "autokeras.StructuredDataClassifier.export_model",
    ],
    "structured_data_regressor.md": [
        "autokeras.StructuredDataRegressor",
        "autokeras.StructuredDataRegressor.fit",
        "autokeras.StructuredDataRegressor.predict",
        "autokeras.StructuredDataRegressor.evaluate",
        "autokeras.StructuredDataRegressor.export_model",
    ],
    "auto_model.md": [
        "autokeras.AutoModel",
        "autokeras.AutoModel.fit",
        "autokeras.AutoModel.predict",
        "autokeras.AutoModel.evaluate",
        "autokeras.AutoModel.export_model",
    ],
    "base.md": [
        "autokeras.Node",
        "autokeras.Block",
        "autokeras.Block.build",
        "autokeras.Head",
    ],
    "node.md": [
        "autokeras.ImageInput",
        "autokeras.Input",
        "autokeras.TextInput",
        "autokeras.StructuredDataInput",
    ],
    "block.md": [
        "autokeras.ConvBlock",
        "autokeras.DenseBlock",
        "autokeras.Embedding",
        "autokeras.Merge",
        "autokeras.ResNetBlock",
        "autokeras.RNNBlock",
        "autokeras.SpatialReduction",
        "autokeras.TemporalReduction",
        "autokeras.XceptionBlock",
        "autokeras.StructuredDataBlock",
        "autokeras.CategoricalToNumerical",
        "autokeras.ImageBlock",
        "autokeras.TextBlock",
        "autokeras.ImageAugmentation",
        "autokeras.Normalization",
        "autokeras.ClassificationHead",
        "autokeras.RegressionHead",
    ],
    "utils.md": [
        "autokeras.image_dataset_from_directory",
        "autokeras.text_dataset_from_directory",
    ],
}


aliases_needed = [
    "tensorflow.keras.callbacks.Callback",
    "tensorflow.keras.losses.Loss",
    "tensorflow.keras.metrics.Metric",
    "tensorflow.data.Dataset",
]


ROOT = "http://autokeras.com/"

autokeras_dir = pathlib.Path(__file__).resolve().parents[1]


def py_to_nb_md(dest_dir):
    dir_path = "py"
    for file_path in os.listdir("py/"):
        file_name = file_path
        py_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != ".py":
            continue

        nb_path = os.path.join("ipynb", file_name_no_ext + ".ipynb")
        md_path = os.path.join(dest_dir, "tutorial", file_name_no_ext + ".md")

        tutobooks.py_to_md(py_path, nb_path, md_path, "templates/img")

        github_repo_dir = "keras-team/autokeras/blob/master/docs/"
        with open(md_path, "r") as md_file:
            button_lines = [
                ":material-link: "
                "[**View in Colab**](https://colab.research.google.com/github/"
                + github_repo_dir
                + "ipynb/"
                + file_name_no_ext
                + ".ipynb"
                + ")   &nbsp; &nbsp;"
                # + '<span class="k-dot">•</span>'
                + ":octicons-mark-github-16: "
                "[**GitHub source**](https://github.com/"
                + github_repo_dir
                + "py/"
                + file_name_no_ext
                + ".py)",
                "\n",
            ]
            md_content = "".join(button_lines) + "\n" + md_file.read()

        with open(md_path, "w") as md_file:
            md_file.write(md_content)


def generate(dest_dir):
    template_dir = autokeras_dir / "docs" / "templates"
    doc_generator = keras_autodoc.DocumentationGenerator(
        PAGES,
        "https://github.com/keras-team/autokeras/blob/master",
        template_dir,
        autokeras_dir / "examples",
        extra_aliases=aliases_needed,
    )
    doc_generator.generate(dest_dir)
    readme = (autokeras_dir / "README.md").read_text()
    index = (template_dir / "index.md").read_text()
    index = index.replace("{{autogenerated}}", readme[readme.find("##") :])
    (dest_dir / "index.md").write_text(index, encoding="utf-8")
    shutil.copyfile(
        autokeras_dir / ".github" / "CONTRIBUTING.md",
        dest_dir / "contributing.md",
    )

    py_to_nb_md(dest_dir)


if __name__ == "__main__":
    generate(autokeras_dir / "docs" / "sources")
