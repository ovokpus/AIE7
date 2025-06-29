import os
from typing import List
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from all pages
        
    Raises:
        ValueError: If there's an error reading the PDF file
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF file {file_path}: {str(e)}")


class PDFFileLoader:
    """Dedicated loader for PDF files with enhanced functionality."""
    
    def __init__(self, path: str):
        self.documents = []
        self.path = path
        self.metadata = []  # Store metadata about each page/document

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        """Load and extract text from a single PDF file."""
        try:
            reader = PdfReader(self.path)
            text = ""
            pages_info = []
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                pages_info.append({
                    "page_number": i + 1,
                    "char_count": len(page_text)
                })
            
            self.documents.append(text.strip())
            self.metadata.append({
                "file_path": self.path,
                "total_pages": len(reader.pages),
                "pages_info": pages_info
            })
            
        except Exception as e:
            raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")

    def load_directory(self):
        """Load all PDF files from a directory."""
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    try:
                        reader = PdfReader(file_path)
                        text = ""
                        pages_info = []
                        
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            text += page_text + "\n"
                            pages_info.append({
                                "page_number": i + 1,
                                "char_count": len(page_text)
                            })
                        
                        self.documents.append(text.strip())
                        self.metadata.append({
                            "file_path": file_path,
                            "total_pages": len(reader.pages),
                            "pages_info": pages_info
                        })
                        
                    except Exception as e:
                        print(f"Warning: Error reading PDF file {file_path}: {str(e)}")

    def load_documents(self):
        """Load documents and return them."""
        self.load()
        return self.documents
    
    def get_metadata(self):
        """Get metadata about loaded PDF files."""
        return self.metadata
    
    def load_pages_separately(self):
        """Load each page as a separate document for more granular processing."""
        if os.path.isfile(self.path) and self.path.endswith(".pdf"):
            try:
                reader = PdfReader(self.path)
                page_documents = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text().strip()
                    if page_text:  # Only add non-empty pages
                        page_documents.append(page_text)
                        self.metadata.append({
                            "file_path": self.path,
                            "page_number": i + 1,
                            "char_count": len(page_text)
                        })
                
                return page_documents
                
            except Exception as e:
                raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")
        else:
            raise ValueError("load_pages_separately only works with single PDF files") 