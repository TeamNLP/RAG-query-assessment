"""
An abstract class representing a framework which can be accumulated.
"""
from typing import Union, Any, Dict, List


class Framework:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def run_gpt(
        self, 
        query: str
    ) -> str:
        assert self.generator.use_hf == False
        
        raise NotImplementedError

    def run_hf(
        self, 
        query: Union[str, Dict[str, Any]]
    ) -> Union[str, List[str]]:
        assert self.generator.use_hf == True
        
        raise NotImplementedError
    
    def run_framework(
        self, 
        query: Union[str, Dict[str, Any]]
    ) -> Union[str, List[str]]:
        if self.generator.use_hf == True:
            output = self.run_hf(query)
        elif self.generator.use_hf == False:
            output = self.run_gpt(query)

        return output
