from unstructured.partition.pdf import partition_pdf

def get_unstructured_auto(pdf_path: str) -> list[dict]:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="auto",
        infer_table_structure=True,
        languages=["eng", "fra"],
        include_page_breaks=True,
    )
    return elements

def get_unstructured_fast(pdf_path: str) -> list[dict]:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",
        infer_table_structure=True,
        languages=["eng", "fra"],
        include_page_breaks=True,
    )
    return elements

def get_unstructured_hires(pdf_path: str) -> list[dict]:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        languages=["eng", "fra"],
        include_page_breaks=True,
    )
    return elements

def get_unstructured_ocr(pdf_path: str) -> list[dict]:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="ocr_only",
        infer_table_structure=True,
        languages=["eng", "fra"],
        include_page_breaks=True,
    )
    return elements
