from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_file_recursively(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Takes a file path and chunks the content recursively.

    Args:
        file_path (str): Path to the text file to be chunked.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(content)

    return chunks

if __name__ == "__main__":
    file_path = 'file.txt'  
    chunks = chunk_file_recursively(file_path)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")
