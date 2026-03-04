import pyttsx3
from langchain_ollama.llms import OllamaLLM as ollama

LLM = ollama(model='llava-phi3')

def speak_text(text):
    new_engine = pyttsx3.init()
    new_engine.setProperty('rate', 170)

    new_engine.say(text)
    new_engine.runAndWait()
    del new_engine 
  
def generate_answer(query):
    try:
        response = LLM.invoke(query)
            
        return response
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Sorry, I couldn't process that."

def main():
    while True:
        user_query = input("You: ")

        if user_query.lower() in ["exit", "quit", "bye"]:
            speak_text("Goodbye!")
            break

        try:
            bot_text = generate_answer(user_query)
            print(f"Bot: {bot_text}")
            speak_text(bot_text)

        except Exception as e:
            print(f"System Error: {e}")


if __name__ == "__main__":
    main()