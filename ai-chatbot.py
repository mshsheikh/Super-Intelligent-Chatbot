from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained DialoGPT-large model and its tokenizer.
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize chat history
chat_history_ids = None

print("Super Intelligent Chatbot (type 'exit' to quit)")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Encode user input and add the end-of-string token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history (if it exists)
    bot_input_ids = new_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # Generate a response while keeping the context (on demand max_length)
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Generated tokens decode in string, no special tokens
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Bot:", bot_response)
