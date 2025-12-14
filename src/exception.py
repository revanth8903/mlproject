
import sys
import logging

# logging.basicConfig(
#     filename="app.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )


def error_message_detail(error, error_detail:sys):   ## Error detail will be present in the "sys"
    _,_,exc_tb = error_detail.exc_info()                  ## exc_info tals about the execution info
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message
 


class CustomException(Exception):
   def __init__(self, error_message, error_detail:sys):
      super().__init__(error_message)
      self.error_message = error_message_detail(error_message, error_detail=error_detail ) ## error_detail is tracked by sys

   def __str__(self):
      return self.error_message
   

# if __name__=="__main__":
   
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by Zero")
#         raise CustomException(e,sys)

    













## exc_tb ---> We will be having "tb.frame"--> "f.code"-->co_filename. It is in the Exception Handling Documentation


## error_detail.exc_info() ---> Will give us three important info, the 1st and 2nd info is not important,
## the 3rd info gives us "exc_tb"

## exc_tb will give the information like on which file the exception has occured, on which line the exception has 
# occured. All those information will be stored in this particular variable.

## [{}] --> PlaceHolder
