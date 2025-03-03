import argparse
import re

from ollama import chat
from pydantic import BaseModel

from pdfextraction.taxonomy import wsl_taxonomy as taxonomy
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
        "--prompt",
        type=str,
        choices=["basic", "main_parts"],
        default=None,
        help="Type of prompt to use for extraction"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name of the json where the extracted information will be stored"
    )

    return parser.parse_args()


def basic_prompt(text: str) -> str:

    prompt = f"""
    You are given a scientific paper. The first page corresponds to the where
    the title, authors, and abstract are located. The rest of the paper is
    divided into sections. Each section has a title and a body. The body of the
    section may contain text, figures, tables, and equations.
    You are tasked with extracting information from the paper.
    Here is the paper:
    {text}
    """

    return prompt


def main_parts_prompt(text: str) -> str:
    # extract the main sections from the pdf
    pattern_parts = r"(\*\*\d+\.(?P<title>[^*]+)\*\*[\s\S]+?)(?=\n\*\*|$)"
    sections = re.findall(pattern_parts, text)

    # find the intro section and the cover page coming before
    title_intro = [title for _, title in sections if "intro" in title.lower()][0]
    cover_page =  text[:text.find(title_intro)]

    # find the conclusion and results part as well
    txt = "\n".join([
        t for t, title in sections if any(
            [m in title.lower() for m in ["results", "conclusion", "intro"]]
        )
    ])

    prompt = f"""
    You are given parts from a scientific paper, you are tasked with
    extracting information from these parts.
    Here is the text:
    {cover_page + txt}
    """
    
    return prompt


def extract_ollama_from_paper(
    prompt: str, model: str, tax: BaseModel
) -> BaseModel:
    
    if not ollama_available(model):
        raise AttributeError(f"{model} is not among the local ollama models.")

    

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

    if args.prompt == "main_parts":
        prompt = main_parts_prompt(txt)
    
    else:
        prompt = basic_prompt(txt)

    paper = extract_ollama_from_paper(prompt, args.model, tax)
    
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
