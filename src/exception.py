import sys #exception ko control krne ke liye sys library chahiye hoti
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()#execution info..ki kis line me kiss file me exception hua h
    file_name=exc_tb.tb_frame.f_code.co_filename#to get file name...custom exception handling google pe search kro vha mil jyega
    error_message="error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    ) #0,1,2 are placeholder
    return error_message

class customexception(Exception): #hm inherit kr rhe exception se
    def __init__(self,error_message,error_detail:sys):#error detail is track ny sys
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
'''if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("divide by zero")
        raise customexception(e,sys)'''