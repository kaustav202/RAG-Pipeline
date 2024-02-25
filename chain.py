from langchain.chains import RetrievalQA  
from langchain.chains import RetrievalQAWithSourcesChain  

class Chain():

    def __init__(self, llm, retriever, prompt="" ) -> None:

        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        
    def qna_chain(self):
        qa = RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=self.retriever,
        chain_type_kwargs={
                "prompt": self.prompt
            }
        )
        return qa

    def qna_sources_chain(self):

        qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(  
            llm=self.llm,  
            chain_type="stuff",  
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )

        return qa_with_sources
