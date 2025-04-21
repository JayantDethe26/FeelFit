document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.querySelector('.chat-input');
    const sendButton = document.querySelector('.send-btn');
    const chatBody = document.querySelector('.chat-body');
    const chips = document.querySelectorAll('.chip');
    
    // Clear existing messages from template
    if (chatBody.children.length > 0) {
        // Keep only the first welcome message
        const welcomeMessage = chatBody.children[0];
        chatBody.innerHTML = '';
        chatBody.appendChild(welcomeMessage);
        
        // Re-create suggestion chips
        const suggestionsDiv = document.createElement('div');
        suggestionsDiv.classList.add('suggestion-chips');
        
        const chipTexts = ['Workout plans', 'Nutrition advice', 'Track progress'];
        chipTexts.forEach(text => {
            const chip = document.createElement('div');
            chip.classList.add('chip');
            chip.textContent = text;
            chip.addEventListener('click', () => {
                sendMessage(text);
            });
            suggestionsDiv.appendChild(chip);
        });
        
        chatBody.appendChild(suggestionsDiv);
    }

    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        
        // Process line breaks and format text
        const formattedText = text.replace(/\n/g, '<br>');
        messageDiv.innerHTML = formattedText;
        
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'bot-message', 'typing-indicator');
        typingDiv.innerHTML = '<span></span><span></span><span></span>';
        typingDiv.id = 'typing-indicator';
        chatBody.appendChild(typingDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async function sendMessage(message) {
        // Don't send empty messages
        if (!message.trim()) return;
        
        addMessage(message, true);
        chatInput.value = '';
        chatInput.focus();
        
        // Show typing indicator
        addTypingIndicator();
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });
            
            // Remove typing indicator
            removeTypingIndicator();
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            if (data.response) {
                addMessage(data.response);
            } else {
                addMessage("Sorry, I couldn't process your request.");
            }
        } catch (error) {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessage("An error occurred. Please try again.");
        }
    }

    sendButton.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (message) sendMessage(message);
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendButton.click();
        }
    });

    // Re-attach event listeners to chips
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            sendMessage(chip.textContent);
        });
    });
});