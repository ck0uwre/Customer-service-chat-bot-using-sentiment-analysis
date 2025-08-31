import os
import json
import requests
from typing import Dict, List, Tuple
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedCustomerServiceBot:
    def __init__(self, aiml_file='customer_service4.aiml'):
        # Initialize AIML Kernel
        try:
            import aiml
            self.kernel = aiml.Kernel()
            self.aiml_file = aiml_file

            # Load or create brain file
            brain_file = 'cs_bot_brain4.brn'
            if os.path.isfile(brain_file):
                self.kernel.bootstrap(brainFile=brain_file)
            elif os.path.isfile(aiml_file):
                self.kernel.bootstrap(learnFiles=aiml_file)
                self.kernel.saveBrain(brain_file)
            else:
                raise FileNotFoundError(f"AIML file '{aiml_file}' is missing.")
        except ImportError:
            raise ImportError("The `aiml` module is not installed. Please install it using pip.")

        # Intent Classification Setup
        self.intent_training_data = self._prepare_intent_training_data()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(
            [phrase for phrases in self.intent_training_data.values() for phrase in phrases]
        )

        # Conversation Management
        self.conversation_history: List[Dict] = []

    def _prepare_intent_training_data(self) -> Dict[str, List[str]]:
        """Prepare training data for intent classification"""
        return {
            'greeting': ['hello there', 'hi how are you', 'good morning', 'greetings'],
            'order_issue': ['my order is late', 'shipping problem', 'tracking not working', 'incorrect item received'],
            'return': ['want to return an item', 'refund request', 'wrong size', 'damaged product'],
            'support': ['need help', 'customer service', 'technical support', 'solve my problem'],
            'billing': ['payment issue', 'charged twice', 'billing question', 'invoice help']
        }

    def classify_intent(self, message: str) -> str:
        """Classify user intent using TF-IDF and cosine similarity"""
        intent_data = [phrase for phrases in self.intent_training_data.values() for phrase in phrases]
        labels = [intent for intent, phrases in self.intent_training_data.items() for _ in phrases]

        input_vector = self.tfidf_vectorizer.transform([message])
        training_matrix = self.tfidf_vectorizer.transform(intent_data)

        similarities = cosine_similarity(input_vector, training_matrix)[0]
        max_similarity_index = similarities.argmax()

        return labels[max_similarity_index]

    def chat_with_gpt(self, prompt: str) -> str:
        """Use Hugging Face API with facebook/blenderbot-400M-distill for generating a response"""
        try:
            API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
            headers = {"Authorization": f"Bearer <YOUR_HUGGING_FACE_API_KEY>"}

            # Add conversational context
            contextual_prompt = f"Customer: {prompt}\nAssistant:"
            payload = {"inputs": contextual_prompt}

            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            response_data = response.json()

            if isinstance(response_data, list) and 'generated_text' in response_data[0]:
                # Extract the generated response
                generated_text = response_data[0]['generated_text'].strip()

                # Remove role prefixes if present
                if generated_text.startswith("Assistant:"):
                    generated_text = generated_text[len("Assistant:"):].strip()

                # Sanitize and return
                return self.sanitize_response(generated_text)
            elif 'error' in response_data:
                return f"Model Error: {response_data['error']}"
            else:
                return "Unexpected response format from Hugging Face API."
        except requests.exceptions.RequestException as req_err:
            return f"Request Error: {req_err}"
        except Exception as e:
            return f"Unexpected Error: {e}"

    def sanitize_response(self, response: str) -> str:
        """Remove inappropriate or nonsensical content from the response"""
        inappropriate_words = ["inappropriate_word", "pedophile", "127.0.0.1", "std::remove"]  # Extend as needed
        sanitized = " ".join(word for word in response.split() if word.lower() not in inappropriate_words)
        if sanitized.strip() == "":
            return "I'm sorry, I couldn't generate a suitable response. Can you rephrase your request?"
        return sanitized

    def _analyze_sentiment(self, text: str) -> float:
        """Perform sentiment analysis using TextBlob"""
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity

        # Adjust sentiment score based on keywords
        negative_keywords = ['refund', 'unhappy', 'bad', 'terrible', 'worst', 'hate', 'problem']
        positive_keywords = ['love', 'great', 'happy', 'good', 'excellent', 'wonderful']

        if any(keyword in text.lower() for keyword in negative_keywords):
            sentiment_score = min(sentiment_score, -0.5)
        elif any(keyword in text.lower() for keyword in positive_keywords):
            sentiment_score = max(sentiment_score, 0.5)

        print(f"Analyzing sentiment for: '{text}' | Sentiment score: {sentiment_score}")
        return sentiment_score

    def get_satisfaction_score(self) -> float:
        """Calculate overall customer satisfaction score"""
        if not self.conversation_history:
            return 0.0

        total_sentiment = sum(entry["sentiment"] for entry in self.conversation_history)
        return total_sentiment / len(self.conversation_history)

    def get_response(self, message: str) -> Tuple[str, float, str]:
        """Generate response for a user message"""
        intent = self.classify_intent(message)
        aiml_response = self.kernel.respond(message)

        # Use Hugging Face API if AIML response is empty or generic
        if not aiml_response or aiml_response.strip() in ["I'm not sure I completely understand. Could you rephrase your question or provide more details?"]:
            aiml_response = self.chat_with_gpt(message)

        # Perform sentiment analysis
        sentiment = self._analyze_sentiment(message)

        # Store the conversation entry
        conversation_entry = {
            'user_input': message,
            'bot_response': aiml_response,
            'sentiment': sentiment,
            'intent': intent
        }
        self.conversation_history.append(conversation_entry)

        return aiml_response, sentiment, intent

    def export_conversation_log(self, filename: str = 'conversation_log.json'):
        """Export conversation history to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        print(f"Conversation log exported to {filename}")

def main():
    bot = AdvancedCustomerServiceBot('customer_service4.aiml')
    print("Smarta: Hello! How can I assist you today?")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye', 'thank you']:
                break
            response, sentiment, intent = bot.get_response(user_input)
            print(f"Smarta(Intent: {intent}, Sentiment: {sentiment:.2f}): {response}")
        except KeyboardInterrupt:
            print("\nConversation ended.")
            break

    print(f"Overall customer satisfaction score: {bot.get_satisfaction_score():.2f}")
    bot.export_conversation_log()

if __name__ == "__main__":
    main()
