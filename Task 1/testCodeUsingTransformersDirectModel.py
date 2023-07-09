import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class QASystem:
    def __init__(self):
        # Load the pre-trained QA model with a larger maximum sequence length# self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad', truncation=True, max_length=512)
        # self.model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad', trust_remote_code=True)

        # the model weights for distil bert mode should already be downloaded in the "./distilbert" directory
        self.tokenizer = AutoTokenizer.from_pretrained('./distilbert', truncation=True, max_length=512)
        self.model = AutoModelForQuestionAnswering.from_pretrained('./distilbert', trust_remote_code=True)

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


qa_system = QASystem()

# ___________Sample ______________
context = """\n\tTransformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other."""
question = "\n\tWhich deep learning libraries back Transformers?"


print("Sample Context: ", context)
print("Sample Question: ", question)

answer = qa_system.extract_answer(question, context)

print("Answer: ", answer)
# ___________Sample ______________

op = input("\nWant to enter a custom query (y/n): ")

if(op == 'y'):
    context = input("Enter the Context: ")
    question = input("Enter the Question: ")

    answer = qa_system.extract_answer(question, context)
    print("answer: ", answer)
