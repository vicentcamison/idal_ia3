from typing import Dict

import numpy as np
import torch
from transformers import BertForQuestionAnswering, BertTokenizer


class QABert:
    def __init__(self):
        """
        Class that encapsulates a Question Answering BERT (Transformers library) for its
        use in inference (prediction). This class does not perform any kind of training
        or fine-tuning.

        """
        model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name) 
        self.model.eval() 
        self.model.to('cpu') 

    def predict(self, question: str, paragraph: str) -> str:
        """
        For a given question and a paragraphs that contains the answer (strings), this 
        function predicts the answer (string) using the BERT-QA model from Transformers 
        Library.
        Args:
            question (`str`): Question that you want to be answered.
            paragraph (`str`): Paragraph where to find the answer.
            
        Returns:
            answer (`str`): Answer that is selected from the paragraph for the given scores

        """
        # Call the method `strings2tensors` to convert question and paragraph into tensors. 
        ###     YOUR CODE HERE      ###
        
        # Use the model to obtain scores.
        ###     YOUR CODE HERE      ###
        
        # Call the method `scores2strings` to convert the scores into a string answer
        ###     YOUR CODE HERE      ###
        return answer
    
    def strings2tensors(self, question: str, paragraph: str) -> Dict[str, torch.Tensor]:
        """
        Converts question and paragraph (strings) into the tensors that BERT needs as 
        inputs. Returns the tensor in a dict.
        Args:
            question (`str`): Question that you want to be answered.
            paragraph (`str`): Paragraph where to find the answer.
            
        Returns:
            bert_args (`dict`): Dictionary where the keys are the arguments that BERT 
            needs as input. Values of the dictionary are the corresponding Torch tensors. 
            {'input_ids': torch.Tensor, 
            'token_type_ids': torch.Tensor}

        """
        # Use the tokenizer to obtain the token ids
        ###     YOUR CODE HERE      ###

        # Construct the list of 0s and 1s for Segment
        ###     YOUR CODE HERE      ###

        # Return the dictionary with the keys `input_ids` and `token_type_ids`
        ###     YOUR CODE HERE      ###
        return bert_args

    def scores2strings(self, 
                       start_score: np.ndarray,
                       end_score: np.ndarray, 
                       input_ids: torch.Tensor) -> str:
        """
        Given the prediction of BERT-QA (start and end scores) and the BERT input IDs, 
        this function transforms the vectors into a string that contains the answer.
        Args:
            start_score (`numpy.ndarray`): Predicted scores of BERT-QA for the start of
                the answer.
            end_score (`numpy.ndarray`): Predicted scores of BERT-QA for the end of the
                answer.
            input_ids (`torch.Tensor`): Tensor that contais the Ids from the tokens that
                forms the BERT input.
            
        Returns:
            answer (`str`): Answer that is selected from the paragraph for the given scores

        """
        # Get the position of the max(start_score) and max(end_score)
        ###     YOUR CODE HERE      ###
        
        # Select the span of token ids given by the positions and convert it into the answer string.
        ###     YOUR CODE HERE      ###
        return answer

    
if __name__ == "__main__":
    QUESTION = 'When were the Normans in Normandy?'
    PARAGRAPH = "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
    QAmodel = QABert()
    answer = QAmodel.predict(QUESTION, PARAGRAPH)
    
    print(f'QUESTION: {QUESTION}\n')
    print(f'PARAGRAPH: {PARAGRAPH}\n')
    print(f'ANSWER: {answer}')


