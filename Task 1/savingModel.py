# Download and save the falcon-7b-instruct model weights:


# from transformers import AutoModelForCausalLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# model_name = "distilbert-base-uncased-distilled-squad"
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model.save_pretrained("distilbert")
# tokenizer.save_pretrained("distilbert")


# This code downloads the model weights from the Hugging Face model hub and saves them in a directory named "falcon-7b-instruct".
#  You can choose any other directory name if you prefer.

# After running the above code, you should have the model weights saved locally and ready for usage in your Falcon application.


model_name = "tiiuae/falcon-7b-instruct"
model = AutoModelForQuestionAnswering.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)


model.save_pretrained("falcon7b")
# tokenizer.save_pretrained("falcon7b")