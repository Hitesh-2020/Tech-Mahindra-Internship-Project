import torch
from waitress import serve
import falcon
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class QASystem:
    def __init__(self):
        # Load the pre-trained QA model
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
        self.model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    def extract_answer(self, question, context):
        # Tokenize the input
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')

        # Get the model predictions
        start_logits, end_logits = self.model(**inputs).values()

        # Find the start and end positions of the answer
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        # Convert the token indices to actual answer
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].tolist()[0])
        answer = ' '.join(tokens[start_index:end_index+1]).replace(' ##', '')

        return answer

class QAResource:
    def __init__(self, qa_system):
        self.qa_system = qa_system

    def on_post(self, req, resp):
        # Get the question and context from the request JSON
        question = req.media['question']
        context = req.media['context']

        # Extract the answer using the QA system
        answer = self.qa_system.extract_answer(question, context)

        # Return the answer or "I don't know"
        resp.media = {'answer': answer if answer else "I don't know"}

# Create the Falcon API
app = falcon.App()

# Initialize the QA system and resource
qa_system = QASystem()
qa_resource = QAResource(qa_system)

# Add the QA resource to the API
app.add_route('/qa', qa_resource)

if __name__ == '__main__':
   serve(app, host='0.0.0.0', port=8000)