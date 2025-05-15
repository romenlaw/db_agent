import customtkinter as ctk
import markdown2
from bot_factory import BotFactory
import tkinter as tk
from tkinter import ttk
import threading
import utils

class ChatBotGUI:
    def __init__(self):
        self.bot = BotFactory.bot()
        
        # Initialize history navigation index
        self.history_index = -1
        
        # Initialize GUI window
        self.window = ctk.CTk()
        self.window.title("RAG Agent")
        self.window.geometry("1000x600")
        
        # Configure grid
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create chat display (using Text widget for markdown rendering)
        self.chat_display = tk.Text(
            self.main_frame,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            font=("Consolas", 11),  # Default monospace font
            bg='#f5f5f5',
            fg='#2b2b2b',
            state='disabled',  # Make text non-editable
            selectbackground='yellow',  # Windows-style selection background
            selectforeground='purple',  # White text for selected content
            exportselection=True  # Allow copying selected text
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.chat_display.tag_configure("user", foreground="#1b5e20", justify="right", font=("Times New Roman", 11))
        self.chat_display.tag_configure("user_bubble", background="#e8f5e9", selectbackground='yellow', spacing1=5, spacing3=5, font=("Times New Roman", 11))
        self.chat_display.tag_configure("bot", foreground="#0d47a1", font=("Consolas", 11))
        self.chat_display.tag_configure("bot_bubble", background="#e3f2fd", selectbackground='yellow', spacing1=5, spacing3=5, font=("Consolas", 11))
                
        # Add scrollbar to chat display
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.chat_display.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_display.configure(yscrollcommand=scrollbar.set)
        
        # Create input frame
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Create controls frame
        controls_frame = ctk.CTkFrame(input_frame)
        controls_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 5))
        controls_frame.grid_columnconfigure(4, weight=1)  # Give weight to temperature slider column
        
        # Add bot type selection
        bot_label = ctk.CTkLabel(controls_frame, text="Bot Type:")
        bot_label.grid(row=0, column=0, padx=(5, 10))
        
        self.bot_combo = ctk.CTkComboBox(
            controls_frame,
            values=BotFactory.available_bots,
            width=200,
            command=self.on_bot_change
        )
        self.bot_combo.set(BotFactory.available_bots[0])  # Set default bot
        self.bot_combo.grid(row=0, column=1, padx=5)
        
        # Add model selection
        model_label = ctk.CTkLabel(controls_frame, text="Model:")
        model_label.grid(row=0, column=2, padx=(20, 10))
        
        # Get available models
        try:
            models = [model['id'] for model in utils.get_available_models()]
        except Exception as e:
            models = [utils.CHAT_MODEL]  # Fallback to default model
            
        self.model_combo = ctk.CTkComboBox(
            controls_frame,
            values=models,
            width=250
        )
        self.model_combo.set(utils.CHAT_MODEL)  # Set default model
        self.model_combo.grid(row=0, column=3, padx=5)
        
        # Add temperature controls
        temp_label = ctk.CTkLabel(controls_frame, text="Temperature:")
        temp_label.grid(row=0, column=4, padx=(20, 10))
        
        self.temp_slider = ctk.CTkSlider(
            controls_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            command=self.update_temp_label
        )
        self.temp_slider.set(0.3)  # Default value
        self.temp_slider.grid(row=0, column=5, sticky="ew", padx=5)
        
        self.temp_value_label = ctk.CTkLabel(controls_frame, text="0.3")
        self.temp_value_label.grid(row=0, column=6, padx=5)
        
        # Add clear history button
        self.clear_button = ctk.CTkButton(
            controls_frame,
            text="Clear History",
            command=self.clear_history,
            width=100
        )
        self.clear_button.grid(row=0, column=7, padx=5)
        
        # Create input field
        self.input_field = ctk.CTkTextbox(input_frame, height=60)
        self.input_field.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Create send button
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=100
        )
        self.send_button.grid(row=1, column=1, padx=5, pady=5)
        
        # Bind Enter key to send message
        self.input_field.bind("<Return>", self.handle_return)
        self.input_field.bind("<Shift-Return>", self.handle_shift_return)
        
        # Bind arrow keys for history navigation
        self.input_field.bind("<Up>", self.handle_up_arrow)
        self.input_field.bind("<Down>", self.handle_down_arrow)
        
        # Initialize loading indicator
        self.loading = False
        self.loading_label = ctk.CTkLabel(input_frame, text="", text_color="red")
        self.loading_label.grid(row=1, column=0, columnspan=2)
        
        # Welcome message
        self.display_bot_message("Welcome! I'm your DARE database assistant. How can I help you today?")

    def handle_return(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # Shift is not pressed
            self.send_message()
            return "break"  # Prevents default behavior
        return None  # Allows default behavior (new line) when Shift is pressed
        
    def get_user_questions(self):
        """Extract user questions from bot's chat history"""
        questions = []
        if hasattr(self.bot, 'chat_history'):
            for message in self.bot.chat_history:
                if message["role"] == "user":
                    questions.append(message["content"])
        # Return questions in reverse order (newest first)
        return list(reversed(questions))
        
    def handle_up_arrow(self, event):
        """Handle Up arrow key press to cycle through chat history"""
        questions = self.get_user_questions()
        
        # Only navigate history if we're at the start of the text or the textbox is empty
        current_text = self.input_field.get("1.0", tk.END).strip()
        cursor_pos = self.input_field.index(tk.INSERT)
        line, column = map(int, cursor_pos.split('.'))
        
        # If we have questions in history
        if questions:
            # If not in history navigation mode yet, start with the first (most recent) question
            if self.history_index == -1:
                self.history_index = 0
            # Otherwise move to older questions if available
            elif self.history_index < len(questions) - 1:
                self.history_index += 1
            
            # Update the input field with the selected question
            self.input_field.delete("1.0", tk.END)
            self.input_field.insert("1.0", questions[self.history_index])
        
        return "break"  # Prevents default behavior
    
    def handle_down_arrow(self, event):
        """Handle Down arrow key press to cycle through chat history"""
        questions = self.get_user_questions()
        
        # Only navigate if we're in history navigation mode
        if self.history_index >= 0:
            # Move to newer questions
            if self.history_index > 0:
                self.history_index -= 1
                self.input_field.delete("1.0", tk.END)
                self.input_field.insert("1.0", questions[self.history_index])
            else:
                # If we're at the most recent question, clear the input field
                self.history_index = -1
                self.input_field.delete("1.0", tk.END)
        
        return "break"  # Prevents default behavior

    def handle_shift_return(self, event):
        """Handle Shift+Enter key press"""
        return None  # Allows default behavior (new line)

    def display_user_message(self, message):
        """Display user message in the chat"""
        self.chat_display.configure(state='normal')  # Temporarily enable for inserting text
        # Add padding to align right
        self.chat_display.insert(tk.END, " " * 50)  # Add space for right alignment
        self.chat_display.insert(tk.END, "You: ", "user")
        
        # Create message bubble
        self.chat_display.insert(tk.END, f"{message}\n", ("user", "user_bubble"))
        self.chat_display.insert(tk.END, "\n")  # Extra space between messages
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')  # Disable again after inserting

    def display_bot_message(self, message):
        """Display bot message in the chat with markdown rendering"""
        self.chat_display.configure(state='normal')  # Temporarily enable for inserting text
        self.chat_display.insert(tk.END, "Bot: ", "bot")
        self.chat_display.insert(tk.END, " ", "bot_bubble")  # Start bubble
        
        # Configure tags for markdown elements
        self.chat_display.tag_configure("code", background="#e3f2fd", foreground="#d32f2f", font=("Consolas", 10))
        self.chat_display.tag_configure("bold", background="#e3f2fd", font=("Consolas", 11, "bold"))
        self.chat_display.tag_configure("italic", background="#e3f2fd", font=("Consolas", 11, "italic"))
        self.chat_display.tag_configure("heading", background="#e3f2fd", font=("Consolas", 13, "bold"))
        
        # Convert markdown to HTML with extras
        html = markdown2.markdown(message, extras=[
            "fenced-code-blocks",
            "tables",
            "code-friendly"
        ])
        
        # Process the message line by line for formatting
        lines = message.split('\n')
        in_code_block = False
        code_block = []
        
        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    code_text = '\n'.join(code_block)
                    self.chat_display.insert(tk.END, code_text + '\n', ("code", "bot_bubble"))
                    code_block = []
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                code_block.append(line)
            else:
                # Process regular text with basic markdown
                if line.startswith('#'):
                    self.chat_display.insert(tk.END, line[1:].strip() + '\n', ("heading", "bot_bubble"))
                else:
                    # Handle bold and italic with bot bubble
                    parts = line.split('**')
                    for i, part in enumerate(parts):
                        if i % 2 == 1:  # Bold text
                            self.chat_display.insert(tk.END, part, ("bold", "bot_bubble"))
                        else:
                            # Handle italic
                            subparts = part.split('*')
                            for j, subpart in enumerate(subparts):
                                if j % 2 == 1:  # Italic text
                                    self.chat_display.insert(tk.END, subpart, ("italic", "bot_bubble"))
                                else:
                                    self.chat_display.insert(tk.END, subpart, "bot_bubble")
                    self.chat_display.insert(tk.END, '\n', "bot_bubble")
        
        self.chat_display.insert(tk.END, '\n')
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')  # Disable again after inserting

    def update_temp_label(self, value):
        """Update temperature value label"""
        self.temp_value_label.configure(text=f"{float(value):.1f}")

    def send_message(self):
        """Send message and get response"""
        message = self.input_field.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Reset history navigation index when sending a new message
        self.history_index = -1
        
        # Clear input field
        self.input_field.delete("1.0", tk.END)
        
        # Display user message
        self.display_user_message(message)
        
        # Disable input while processing
        self.input_field.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.loading_label.configure(text="Thinking...")
        
        # Get response in a separate thread with current temperature
        temperature = float(self.temp_slider.get())
        threading.Thread(target=self.get_response, args=(message, temperature), daemon=True).start()

    def get_response(self, message, temperature):
        """Get response from the chat bot"""
        try:
            model = self.model_combo.get()
            response = self.bot.chat(message, model=model, temperature=temperature)
            
            # Update GUI in the main thread
            self.window.after(0, self.handle_response, response)
        except Exception as e:
            self.window.after(0, self.handle_error, str(e))
        finally:
            # Re-enable input
            self.window.after(0, self.enable_input)

    def handle_response(self, response):
        """Handle bot response"""
        self.display_bot_message(response)

    def handle_error(self, error_message):
        """Handle error in chat"""
        self.display_bot_message(f"Error: {error_message}")

    def enable_input(self):
        """Re-enable input field and button"""
        self.input_field.configure(state="normal")
        self.send_button.configure(state="normal")
        self.loading_label.configure(text="")

    def on_bot_change(self, choice):
        """Handle bot type change"""
        self.bot = BotFactory.bot(choice)
        self.clear_history()

    def clear_history(self):
        """Clear chat history and display"""
        # Clear display
        self.chat_display.configure(state='normal')
        self.chat_display.delete('1.0', tk.END)
        self.chat_display.configure(state='disabled')
        
        # Create new bot instance with same type to clear history
        current_bot = self.bot_combo.get()
        # self.bot = BotFactory.bot(current_bot)
        self.bot.new_chat()
        
        # Reset history navigation index
        self.history_index = -1
        
        # Display welcome message again
        # welcome_messages = {
        #     'DARE expert': "Welcome! I'm your DARE database assistant. How can I help you today?",
        #     'Interchange Fee expert': "Welcome! I'm your Interchange Fee expert. How can I help you today?"
        # }
        # self.display_bot_message(welcome_messages.get(current_bot, "Welcome! How can I help you today?"))
        self.display_bot_message(BotFactory._config[current_bot]["greeting"])
    def run(self):
        """Start the GUI application"""
        self.window.mainloop()

if __name__ == "__main__":
    print('Initializing chat bot, please wait...')
    chat_gui = ChatBotGUI()
    chat_gui.run()
