from prompt_toolkit.key_binding import KeyBindings
kb = KeyBindings()

@kb.add('enter')
def _(event):
    import logging
    logger = logging.getLogger(__name__)

    buffer = event.app.current_buffer
    logger.debug("Current buffer state: %s", buffer)

    if buffer.complete_state:
        current_completion = buffer.complete_state.current_completion
        logger.debug("Autocomplete menu is open. Current completion: %s", current_completion)
        if current_completion:
            # If the autocomplete menu is open and there's a current completion, select it
            buffer.apply_completion(current_completion)
            logger.debug("Applied completion: %s", current_completion)
    else:
        # If autocomplete menu is not open, submit the message
        buffer.validate_and_handle()
        logger.debug("Autocomplete menu is not open. Message submitted.")
