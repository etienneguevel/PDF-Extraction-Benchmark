# README

This github repo covers the extraction of information from the articles used for the
Data for Good project led by the sufficiency lab.  

## Install

The package is managed with `uv` but you can simply install it by creating a python virtual environment,  
and then call:

`pip install .`

The extraction of the taxonomy as of now needs `ollama >= 0.5.12`, be sure to get the last version of [ollama](https://ollama.com/download/linux).

## Usage

In the `src` different folders cover the usage cases:

- scripts for the conversion pdfs -> texts are in the `pdf_extraction` folder
- scripts for the extraction of information from texts are in the `llm_extraction` folder
- objects for the taxonomy are in the `taxonomy` folder
