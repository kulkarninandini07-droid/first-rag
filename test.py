print("Testing imports...")

import torch
print(f"✅ PyTorch: {torch.__version__}")

from sentence_transformers import SentenceTransformer
print("✅ Sentence Transformers working")

import faiss
print("✅ FAISS working")

import pdfplumber
print("✅ pdfplumber working")

import numpy as np
print("✅ NumPy working")

print("\n🎉 Everything is set up! You're ready to build.")