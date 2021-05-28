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
        # Convert question and paragraph into tensors. Input that BERT needs.
        bert_inputs = self.strings2tensors(question, paragraph)
        
        # Use the model for prediction. Calculate the scores
        with torch.no_grad():
            scores = self.model(**bert_inputs)
        
        # Convert the scores to a string answer
        answer = self.scores2strings(start_score=scores.start_logits.cpu().numpy().squeeze(), 
                                     end_score=scores.end_logits.cpu().numpy().squeeze(), 
                                     input_ids=bert_inputs['input_ids'])
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
        input_ids = self.tokenizer.encode(question, paragraph)

        # Construct the list of 0s and 1s for Segment
        sep_index = input_ids.index(self.tokenizer.sep_token_id)
        segment_ids = [0]*(sep_index + 1) + [1]*(len(input_ids) - sep_index - 1)
        assert len(segment_ids) == len(input_ids)

        bert_args = {'input_ids': torch.tensor([input_ids]).to('cpu'), 
                     'token_type_ids': torch.tensor([segment_ids]).to('cpu')}
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
        # Get the postion of the max score in start and end
        answer_start, answer_end = np.argmax(start_score), np.argmax(end_score)
        answer_ids = input_ids[0][answer_start:answer_end + 1]
        answer = self.tokenizer.decode(answer_ids, 
                                       skip_special_tokens=True, 
                                       clean_up_tokenization_spaces=True)
        return answer

    
if __name__ == "__main__":
    QUESTION = 'When were the Normans in Normandy?'
    PARAGRAPH = "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
    QAmodel = QABert()
    answer = QAmodel.predict(QUESTION, PARAGRAPH)
    
    print(f'QUESTION: {QUESTION}\n')
    print(f'PARAGRAPH: {PARAGRAPH}\n')
    print(f'ANSWER: {answer}')


