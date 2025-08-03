import sys

def error_message_detail(err, err_detail: sys) -> str:
    """
    This function takes the error and returns a custom error message.
    :param err:
    :param err_detail:
    :return:
    """

    _, _,exc_tb = err_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    err_msg = ("Error Occurred in Python Script\n"
               "Name:            [{0}]\n"
               "Line Number:     [{1}]\n"
               "Error Message:   [{2}]").format(file_name, exc_tb.tb_lineno, str(err))

    return err_msg

class CustomException(Exception):
    def __init__(self, err_msg, err_detail: sys):
        super().__init__(err_msg)

        self.err_msg = error_message_detail(err_msg, err_detail)

    def __str__(self):
        return self.err_msg
