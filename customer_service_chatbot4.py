import os
import json
import requests
from typing import Dict, List, Tuple
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

class AdvancedCustomerServiceBot:
    def __init__(self, aiml_file='customer_service4.aiml'):
        # Initialize AIML Kernel
        try:
            # Python 3.8+ removed time.clock; aiml library still uses it. Monkey-patch to avoid AttributeError.
            import time
            if not hasattr(time, 'clock'):
                time.clock = time.perf_counter

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
        try:
            client = InferenceClient(
                api_key="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"
            )

            system_msg = (
                "You are a professional and polite customer service assistant. "
                "Provide concise and relevant answers, avoiding unnecessary or random sentences. "
                "Keep responses brief (one or two sentences) unless more detail is requested."
            )

            completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2:together",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )

            raw_resp = completion.choices[0].message.content.strip()
            # truncate after two sentences to avoid overly long replies
            sentences = raw_resp.split('.')
            if len(sentences) > 2:
                truncated = '.'.join(sentences[:2]).strip()
                if not truncated.endswith('.'):
                    truncated += '.'
            else:
                truncated = raw_resp

            return self.sanitize_response(truncated)

        except Exception as e:
            return f"LLM Error: {str(e)}"

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
        negative_keywords = ['refund', 'unhappy', 'bad', 'terrible', 'worst', 'hate', 'problem', 'issue', 'late', 'wrong', 'damaged']
        positive_keywords = ['love', 'great', 'happy', 'good', 'excellent', 'wonderful', 'satisfied',]

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
        # handle blank messages gracefully
        if not message or not message.strip():
            return (
                "I'm here whenever you're ready—feel free to type your question or issue.",
                0.0,
                "none"
            )

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
