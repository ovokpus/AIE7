import os
from typing import List

# Import PDF functionality from the dedicated module
try:
    from .pdf_utils import extract_text_from_pdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_pdf_file()
        else:
            supported_types = ".txt"
            if PDF_SUPPORT:
                supported_types += " or .pdf"
            raise ValueError(
                f"Provided path is neither a valid directory nor a supported file type ({supported_types})."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file(self):
        """Load and extract text from a PDF file."""
        if not PDF_SUPPORT:
            raise ValueError("PDF support not available. Please install pypdf: pip install pypdf")
        
        try:
            text = extract_text_from_pdf(self.path)
            self.documents.append(text)
        except Exception as e:
            raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding=self.encoding) as f:
                        self.documents.append(f.read())
                elif file.endswith(".pdf") and PDF_SUPPORT:
                    try:
                        text = extract_text_from_pdf(file_path)
                        self.documents.append(text)
                    except Exception as e:
                        print(f"Warning: Error reading PDF file {file_path}: {str(e)}")
                elif file.endswith(".pdf") and not PDF_SUPPORT:
                    print(f"Warning: Skipping PDF file {file_path} - PDF support not available")

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks
