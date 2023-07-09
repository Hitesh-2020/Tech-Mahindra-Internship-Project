from transformers import pipeline

# ___________Sample ______________
context = """\n\tTransformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other."""
question = "\n\tWhich deep learning libraries back Transformers?"


print("Sample Context: ", context)
print("\nSample Question: ", question)

# Replace this with your own checkpoint

model_checkpoint = "deepset/roberta-base-squad2"
# model_checkpoint = "./distilbert"
question_answerer = pipeline("question-answering", model=model_checkpoint)


answer = question_answerer(question=question, context=context)

print("\nAnswer: ", answer['answer'])
# ___________Sample ______________

op = input("\n\nWant to enter a custom query (y/n): ")

if(op == 'y'):
    context = input("\nEnter the Context: ")
    question = input("\nEnter the Question: ")

    # model_checkpoint = "deepset/roberta-base-squad2"
    model_checkpoint = "./distilbert"
    question_answerer = pipeline("question-answering", model=model_checkpoint)


    answer = question_answerer(question=question, context=context)
    print("\nAnswer: ", answer['answer'])