import argparse

from ollama import chat
from pydantic import BaseModel

import pdfextraction.taxonomy.wsl_taxonomy as taxonomy
from pdfextraction.llm.utils import open_file, ollama_available

def parse_args():
    parser = argparse.ArgumentParser(description="Extract OLLAMA from a paper")
    parser.add_argument("--text-path", type=str, help="Path to the text file")
    parser.add_argument("--model", type=str, help="Model to use for extraction")
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=None,
        help="Taxonomy to use for extraction"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name of the json where the extracted information will be stored"
    )

    return parser.parse_args()


def extract_ollama_from_paper(
    text: str, model: str, tax: BaseModel
) -> BaseModel:
    
    if not ollama_available(model):
        raise AttributeError(f"{model} is not among the local ollama models.")

    prompt = f"""
    You are given a scientific paper. The first page corresponds to the where
    the title, authors, and abstract are located. The rest of the paper is
    divided into sections. Each section has a title and a body. The body of the
    section may contain text, figures, tables, and equations.
    You are tasked with extracting information from the paper.
    Here is the paper:
    {text}
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

    paper = tax.model_validate_json(response.message.content)

    return paper

def main():
    args = parse_args()

    # check if the given taxonomy name is within the coded taxonomies
    if args.taxonomy:
        callables = {
            name: obj for name, obj in taxonomy.__dict__.items() if callable(obj)
        }
        try:
            tax = callables[args.taxonomy]

        except KeyError:
            raise ValueError(f"Taxonomy {args.taxonomy} not found")

    # if no given taxonomies, use the default one   
    else:
        tax = taxonomy.Paper

    # open and process the text
    txt = open_file(args.text_path)
    paper = extract_ollama_from_paper(txt, args.model, tax)
    
    # print or save the output
    if not args.output_path:
        print(paper)
    
    # save the result with the given name
    else:
        output_path = args.output_path.split(".")[0] + ".json"
        with open(output_path, "w") as f:
            f.write(paper.model_dump_json())

if __name__ == "__main__":
    main()
