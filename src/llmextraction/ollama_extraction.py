import argparse

from ollama import chat

import src.taxonomy.wsl_taxonomy as taxonomy


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
        "--output-path",
        type=str,
        default=None,
        help="Path to save the extracted OLLAMA"
    )

    return parser.parse_args()

def open_file(text_path):
    with open(text_path, "rb") as f:
        return f.read()
    
    return text_path

def extract_ollama_from_paper(text_path, model, tax):

    text = open_file(text_path)
    response = chat(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
        model=model,
        format=tax.model_json_schema()
    )

    paper = tax.model_validate_json(response.message.content)

    return paper

def main():
    args = parse_args()
    if args.taxonomy:
        callables = {
            name: obj for name, obj in taxonomy.__dict__.items() if callable(obj)
        }
        try:
            tax = callables[args.taxonomy]

        except KeyError:
            raise ValueError(f"Taxonomy {args.taxonomy} not found")
        
    else:
        tax = taxonomy.Paper

    paper = extract_ollama_from_paper(args.text_path, args.model, tax)
    
    if not args.output_path:
        print(paper)
    
    else:
        output_path = args.output_path.split(".")[0] + ".json"
        with open(output_path, "w") as f:
            f.write(paper.model_dump_json())

if __name__ == "__main__":
    main()
