import itertools
import re

from sphinx.util.typing import stringify_annotation

from . import utils


def get_code_blocks(docstring):
    code_blocks = {}
    tmp = docstring[:]
    while "```" in tmp:
        tmp = tmp[tmp.find("```") :]
        index = tmp[3:].find("```") + 6
        snippet = tmp[:index]
        # Place marker in docstring for later reinjection.
        token = f"$KERAS_AUTODOC_CODE_BLOCK_{len(code_blocks)}"
        docstring = docstring.replace(snippet, token)
        code_blocks[token] = snippet
        tmp = tmp[index:]

    return code_blocks, docstring


def get_section_end(docstring, section_start):
    regex_indented_sections_end = re.compile(r"\S\n+(\S|$)")
    end = re.search(regex_indented_sections_end, docstring[section_start:])
    section_end = section_start + end.end()
    if section_end == len(docstring):
        return section_end
    else:
        return section_end - 2


def get_google_style_sections_without_code(docstring):
    regex_indented_sections_start = re.compile(r"\n# .+?\n")

    google_style_sections = {}
    for i in itertools.count():
        match = re.search(regex_indented_sections_start, docstring)
        if match is None:
            break
        section_start = match.start() + 1
        section_end = get_section_end(docstring, section_start)
        google_style_section = docstring[section_start:section_end]
        token = f"KERAS_AUTODOC_GOOGLE_STYLE_SECTION_{i}"
        google_style_sections[token] = google_style_section
        docstring = utils.insert_in_string(
            docstring, token, section_start, section_end
        )
    return google_style_sections, docstring


def get_google_style_sections(docstring):
    # First, extract code blocks and process them.
    # The parsing is easier if the #, : and other symbols aren't there.
    code_blocks, docstring = get_code_blocks(docstring)

    google_style_sections, docstring = get_google_style_sections_without_code(
        docstring
    )

    docstring = reinject_strings(docstring, code_blocks)
    for section_token, section in google_style_sections.items():
        google_style_sections[section_token] = reinject_strings(
            section, code_blocks
        )
    return google_style_sections, docstring


def to_markdown(
    google_style_section: str, types: dict = None, aliases=None
) -> str:
    end_first_line = google_style_section.find("\n")
    section_title = google_style_section[2:end_first_line]
    section_body = google_style_section[end_first_line + 1 :]
    section_body = utils.remove_indentation(section_body.strip())

    # it's a list of elements, a special formatting is applied.
    if section_title == "Arguments":
        section_body = format_as_markdown_list(section_body, types, aliases)
    elif section_title in ("Attributes", "Raises"):
        section_body = format_as_markdown_list(section_body)

    if section_body:
        return f"__{section_title}__\n\n{section_body}\n"
    else:
        return f"__{section_title}__\n"


def format_as_markdown_list(
    section_body, types: dict = None, aliases: dict = None
):
    section_body = re.sub(r"\n([^ ].*?):", r"\n- __\1__:", section_body)
    section_body = re.sub(r"^([^ ].*?):", r"- __\1__:", section_body)

    # Optionally add type annotations to docstring
    if types:
        for arg, arg_type in types.items():
            type_hint_str = apply_aliases(
                stringify_annotation(arg_type), aliases
            )
            section_body = re.sub(
                rf"(- __{arg}__)", rf"\1 `{type_hint_str}`", section_body
            )

    return section_body


def apply_aliases(string: str, aliases: dict):
    for dotted_path, alias in aliases.items():
        string = string.replace(dotted_path, alias)
    return string


def reinject_strings(target, strings_to_inject):
    for token, string_to_inject in strings_to_inject.items():
        target = target.replace(token, string_to_inject)
    return target


def process_docstring(docstring, types: dict = None, aliases=None):
    if docstring[-1] != "\n":
        docstring += "\n"
    google_style_sections, docstring = get_google_style_sections(docstring)

    for token, google_style_section in google_style_sections.items():
        markdown_section = to_markdown(google_style_section, types, aliases)
        docstring = docstring.replace(token, markdown_section)
    return docstring
