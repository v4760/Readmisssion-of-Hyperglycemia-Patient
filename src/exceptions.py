import sys
from logger import logging

def error_message_details(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()

    error_message = 'Error Occoured!\n Details:\nFile name: [{0}]\nLine number[{1}]\nError details[{2}]'.format(
        exc_tb.tb_frame.f_code.co_filename,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_message
class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self):
        return self.error_message

if __name__ == '__main__':
    try:
        a= 1/0
    except Exception as e:
        logging.info('Trying to divide by zero')
        raise CustomException(e,sys)
