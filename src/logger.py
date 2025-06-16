#any execution which will happen we should able to log tha
import logging
import os
from datetime import datetime

log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#Iska use hota hai logging files banane ke liye taaki har run pe nayi file bane.
#Aise file ka naam unique hota hai, toh purane logs overwrite nahi hote.
logs_path=os.path.join(os.getcwd(),"logs",log_file)
#logs_path ek variable hai jisme log file ka full path banaya jaa raha hai. Jaise:  C:/Users/shwet/your_project/logs/16_06_2025_18_03_55.log
os.makedirs(logs_path,exist_ok=True)#Yeh command ek folder ya folders create karti hai.Agar wo folder already exist karta hai, to koi error nahi degi.

log_file_path=os.path.join(logs_path,log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

'''if __name__=="__main__":
    logging.info("logging has started")'''