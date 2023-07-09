import falcon
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class QASystem:
    def __init__(self):
        # Load the pre-trained QA model with a larger maximum sequence length
        self.tokenizer = AutoTokenizer.from_pretrained('./distilbert', truncation=True, max_length=512)
        self.model = AutoModelForQuestionAnswering.from_pretrained('./distilbert')

    def extract_answer(self, question, context):
        # Tokenize the input with the larger maximum sequence length and truncation enabled
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, max_length=512, truncation=True, return_tensors='pt')

        # Get the model predictions
        start_logits, end_logits = self.model(**inputs).values()

        # Find the start and end positions of the answer
        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item()

        # Convert the token indices to actual answer
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].tolist()[0])
        answer = ' '.join(tokens[start_index:end_index+1]).replace(' ##', '')

        return answer

class QAResource:
    def __init__(self, qa_system):
        self.qa_system = qa_system

    def on_post(self, req, resp):
        # Get the query from the request JSON
        query = req.media['query']

        # Extract the question and context from the query
        question, context = self.extract_question_context(query)

        # Extract the answer using the QA system
        answer = self.qa_system.extract_answer(question, context)

        # Return the answer
        resp.media = {'answer': answer}

    def extract_question_context(self, query):
        # Split the query into question and context based on the delimiter ":"
        parts = query.split(":", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        else:
            return None, None

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
