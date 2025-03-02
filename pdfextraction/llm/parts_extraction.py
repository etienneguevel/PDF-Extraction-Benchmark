import argparse

from .ollama_extraction import extract_ollama_from_paper
from .parts import parts

from ollama import chat
import pdfextraction.taxonomy.wsl_taxonomy as taxonomy



def parse_args():
    parser = argparse.ArgumentParser(description="Extract OLLAMA from a list of paragraphs")
    parser.add_argument("--model", type=str, help="Model to use for extraction")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the extracted OLLAMA"
    )

    return parser.parse_args()


def extract_info_from_list(l, model, tax):
    output = []
    for i in l:

        prompt = f"""
        You are given the paragraph of a scientific paper
        You are tasked with extracting information from this paragraph.
        Here is the text:
        {i}
        """

        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            format=tax.model_json_schema()
        )
        out = tax.model_validate_json(response.message.content)
        output.append(out)

    unifying_prompt = f"""
    You are given a list of versions of taxonomies, each one corresponding to a paragraph of a scientific paper.
    You are tasked with unifying this information into a single taxonomy, using the given text.
    Give the most solid taxonomy.
    Here are the taxonomies:
    {output}

    Here is the text:
    {'\n'.join(l)}

    """

    response = chat(
        messages=[
            {
                "role": "user",
                "content": unifying_prompt,
            }
        ],
        model=model,
        format=tax.model_json_schema()
    )
    result = tax.model_validate_json(response.message.content)

    return result




def main():

    args = parse_args()
    output = extract_info_from_list(
        parts,
        args.model,
        taxonomy
    )

    if not args.output_path:
        print(output)

    else:
        with open(args.output_path, "w") as f:
            f.write(output)

if __name__ == "__main__":
    main()
