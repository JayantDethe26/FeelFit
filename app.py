from flask import Flask, render_template, request, jsonify, session
from fit import chat_with_feelfit, retriever, llm, embeddings, ChatMessageHistory, profile_manager
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Initialize components from fit.py
if not os.path.exists("datatext.txt"):
    # Run PDF processing if needed
    from fit import process_pdfs
    # Uncomment to force processing: process_pdfs()

@app.route('/')
def chatbot():
    # Initialize session if new user
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_input = data['message']
        
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Get or initialize chat history for session
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Convert session history to ChatMessageHistory
        chat_history = ChatMessageHistory()
        for msg in session['chat_history']:
            if msg['type'] == 'human':
                chat_history.add_user_message(msg['content'])
            else:
                chat_history.add_ai_message(msg['content'])
        
        # Process message with session ID for user tracking
        response = chat_with_feelfit(user_input, chat_history, session_id)
        
        # Store updated history in session
        session['chat_history'] = [
            {'type': msg.type, 'content': msg.content}
            for msg in chat_history.messages
        ]
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error in /ask route: {str(e)}")
        return jsonify({'response': f"I'm having trouble processing your request right now. Could you try again?"}), 500

# Optional debug route to check user profile
@app.route('/debug/profile', methods=['GET'])
def debug_profile():
    if app.debug:  # Only available in debug mode
        session_id = session.get('session_id', 'default')
        profile = profile_manager.get_profile(session_id)
        return jsonify(profile)
    return "Not available", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

