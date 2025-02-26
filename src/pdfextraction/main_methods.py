# Description: This file contains the main methods used in the program.
from PyPDF2 import PdfReader
import datetime

def get_pdf_info(pdf_path: str) -> dict:
    """
    Get the information of a PDF file.

    :param pdf_path: Path to the PDF file.
    :return: Dictionary with the information of the PDF file.
    """
    pdf_info = {}
    reader = PdfReader(pdf_path)
    meta = reader.metadata

    pdf_info["title"] = meta.title
    pdf_info["author"] = meta.author
    pdf_info["subject"] = meta.subject
    pdf_info["producer"] = meta.producer
    pdf_info["creator"] = meta.creator
    pdf_info["pages"] = len(reader.pages)

    return pdf_info


def extract_and_measure_timing(func, *args, **kwargs):
    """
    Extract content from a PDF file and measure the timing.

    :param func: Function to extract content from a PDF file.
    :param args: Positional arguments for the function.
    :param kwargs: Keyword arguments for the function.
    :return: Content extracted and timing information.
    """
    start_time = datetime.datetime.now()
    content = func(*args, **kwargs)
    end_time = datetime.datetime.now()
    timing = end_time - start_time
    return content, timing.total_seconds()
