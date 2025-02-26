# %% [markdown]
# # Imports

# %%
# %pip uninstall -y pytesseract

# %%
# %pip install -r requirements.txt

# %%
%pip install -q -r requirements.txt

# %% [markdown]
# ## Lib imports

# %%
import matplotlib.pyplot as plt
from icecream import ic
import seaborn as sns
import pandas as pd
from ollama import Client as OllamaClient
from pydantic import BaseModel

import platform
import os

# %% [markdown]
# ## Methods imports

# %%
from main_methods import (
    get_pdf_info,
    extract_and_measure_timing,
)

## PDF Extraction Methods
from extraction_methods.pymupdf4llm import get_pymupdf4llm
from extraction_methods.unstructured import (
    get_unstructured_auto,
    get_unstructured_fast,
    get_unstructured_hires,
    get_unstructured_ocr,
)

# %% [markdown]
# ## WSL Taxonomy classes

# %%
from wsl_taxonomy import *

# %% [markdown]
# # Functions

# %% [markdown]
# ## OS and other tools

# %%
def get_path_of_all_files_in_subfolders_with_extension_in_list(
    folder_path: str, extensions: list[str], ignore_folders: list[str] = []
) -> list:
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        # print('root', root)
        # print('dirs', dirs)
        # print('files', files)
        if any([(os.sep + ignore_folder) in root for ignore_folder in ignore_folders]):
            continue
        # if root.__contains__(r"\outputs"):
        #     continue

        for file in files:
            for extension in extensions:
                if file.lower().endswith(extension) and not file.startswith("."):
                    all_files.append(os.path.join(root, file))
    return all_files


# %%
def check_if_each_parent_folders_exists_and_if_not_create_it(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# %%
def call_ollama_server(used_model, system_prompt, user_prompt):
    local_client = OllamaClient(host="http://localhost:11434")
    response = local_client.chat(
        model=used_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )
    return response

# %%
def ask_LLM_to_choose_main_author(text_content):
    my_model = "llama3.2"
    my_system_prompt = """You are a talented, organised and detail oriented researcher.
                        You have been given a study to review and your main task is to classify some information regarding the user prompt.
                        Here you're concern is to extract the main author of the study, and his/her genra.
                        **The answer should be only the name of a single person and his/her genra.**
                     """
    ## Create user prompt in english
    my_user_prompt = f"""{text_content}"""

    response = call_ollama_server(my_model, my_system_prompt, my_user_prompt)
    return_resp = response["message"]["content"]

    return return_resp


# %%
ask_LLM_to_choose_main_author(all_text)

# %% [markdown]
# # Main run

# %%
pdf_list = get_path_of_all_files_in_subfolders_with_extension_in_list(
    "sources_pdf", [".pdf"], ["outputs"]
)
print(f"Found {len(pdf_list)} pdf files")

# %%
## Get the hash of the name of the host machine
host_name = hash(platform.node())

# %%
# Create a dataframe to store the data and timing of processing the pdf files
df_pdf_processing = pd.DataFrame(
    columns=[
        "pdf_path",
        "pdf_name",
        "pages_length",
        "file_size",
        "method",
        "output_md",
        "time_processing",
        "compute_identifier",
    ]
)

# %%
### Define methods list to be used
methods_list = [
    get_pymupdf4llm,
    # get_unstructured_auto,
    get_unstructured_fast,
    get_unstructured_hires,
    # get_unstructured_ocr,
]

# %%
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# %%
# for pdf_path in pdf_list[:3]:
for pdf_path in pdf_list:
    pdf_name = os.path.basename(pdf_path)
    ic(f"Processing {pdf_name}")
    file_info = get_pdf_info(pdf_path)
    file_length = file_info["pages"]

    file_size = os.path.getsize(pdf_path) / 1024 / 1024

    for method in methods_list:
        ic(method.__name__)  # Print the method name
        # Extract and measure the time of the method
        output = extract_and_measure_timing(
            method,
            pdf_path=pdf_path,
            # bool_write_images=False,
            # bool_embed_images=False,
        )
        # Append the data to the dataframe
        df_pdf_processing.loc[len(df_pdf_processing)] = [
            pdf_path,
            pdf_name,
            file_length,
            file_size,
            method.__name__,
            output[0],
            output[1],
            host_name,
        ]

# %%
df_pdf_processing.shape

# %%
df_pdf_processing.head(2)

# %% [markdown]
# ## TODO : Store the output of each methods

# %%
## Force an object to a string


# %%
pymupdf4llm_output = df_pdf_processing[
    (df_pdf_processing["method"] == "get_pymupdf4llm")
    & (df_pdf_processing["pdf_name"] == "2000gollan038.pdf")
    # (df_pdf_processing["pdf_name"] == "2000gollan038.pdf")
]["output_md"].values[0]

# ["output_md"]

# get_pymupdf4llm
# get_unstructured_fast

# %%
pymupdf4llm_output

# %%
all_text = ""
for atext in pymupdf4llm_output:
    all_text += atext["text"] + "\n\n"

# %%
all_text

# %%
### Write the output to a file
with open("o2000gollan038_pymupdf4llm.txt", "w") as text_file:
    text_file.write(all_text)

# %%
## For each row, export the output to a markdown file in the outputs folder : extracted_outputs\methods_outputs
current_folder = os.getcwd()
for index, row in df_pdf_processing.iterrows():
    output_folder_name = row["pdf_name"].replace(".pdf", "")
    row_folder = os.path.join(
        current_folder, "extracted_outputs", "methods_outputs", output_folder_name
    )
    row_folder_checked = check_if_each_parent_folders_exists_and_if_not_create_it(
        row_folder
    )
    row_output_file_path = os.path.join(row_folder_checked, row["method"] + ".md")

    with open(row_output_file_path, "w") as text_file:
        ic(row["pdf_name"])
        try:
            text_file.write(str(row["output_md"]))
        except Exception as e:
            ic(e.__context__)
        #     ic(type(row["output_md"]))
        #     # ic(row["output_md"])


# %%
## Hash all the values from that hash column into one unique value, not a dataframe column
hash_all = hash(
    tuple(df_pdf_processing["pdf_path"]) + tuple(df_pdf_processing["method"])
)

## Exclude the column with the output_md
df_export = df_pdf_processing.drop(columns=["output_md"])
df_export.to_csv(
    os.path.join(
        "extracted_outputs", "timing_dfs", f"df_pdf_processing_{hash_all}.csv"
    ),
    index=False,
)

# %%


# %% [markdown]
# ## Plot the timing of each methods

# %%
# Print a graph, with file_size on the x-axis and time taken to process the pdf on the y-axis, spliting the data in series by the method used to process the pdf
sns.scatterplot(
    data=df_pdf_processing,
    # x="file_size"
    x="pages_length",
    y="time_processing",
    size="file_size",
    hue="method",
)
plt.show()

# %% [markdown]
# # Content Analysis

# %%
##TODO


