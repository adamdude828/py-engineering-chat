from prompt_toolkit.key_binding import KeyBindings
kb = KeyBindings()

@kb.add('enter')
def _(event):
    buffer = event.app.current_buffer
    if buffer.complete_state:
        # If the autocomplete menu is open, select the current completion
        buffer.apply_completion(buffer.complete_state.current_completion)
    else:
        # If autocomplete menu is not open, submit the message
        buffer.validate_and_handle()
