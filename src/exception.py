from typing import Optional

def error_message_detail(err: BaseException) -> str:
    """
    Build a readable error message from an exception, including file and line.
    """
    tb = err.__traceback__
    # Walk to the last frame (where the error actually occurred)
    while tb and tb.tb_next:
        tb = tb.tb_next

    file_name = tb.tb_frame.f_code.co_filename if tb else "<unknown>"
    line_no = tb.tb_lineno if tb else -1
    return (
        "Error Occurred in Python Script\n"
        f"Name:            [{file_name}]\n"
        f"Line Number:     [{line_no}]\n"
        f"Error Message:   [{err}]"
    )

class CustomException(Exception):
    """
    Wraps an underlying exception with nicer context while preserving the cause.
    """
    def __init__(self, message: str, *, cause: Optional[BaseException] = None):
        self.message = message
        self.cause = cause
        detail = error_message_detail(cause) if cause else message
        super().__init__(f"{message}\n{detail}" if cause else message)

    def __str__(self) -> str:
        return self.args[0]
