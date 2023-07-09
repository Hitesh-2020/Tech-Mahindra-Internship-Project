import torch
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

        # Check if the predicted answer is valid
        if start_index.item() > end_index.item():
            return "I don't know my lord!"

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
        resp.media = {'answer': answer}

# Create an instance of the Falcon App
app = falcon.App()

# Initialize the QA system
qa_system = QASystem()

# Add the QA resource to the app
qa_resource = QAResource(qa_system)
app.add_route('/qa', qa_resource)

if __name__ == '__main__':
    # Run the app on localhost:8000
    from wsgiref import simple_server

    httpd = simple_server.make_server('127.0.0.1', 8000, app)
    httpd.serve_forever()
