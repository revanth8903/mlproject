
import logging
import os
from datetime import datetime

LOG_FILE = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(logs_path, exist_ok=True)  ## Even though there is a file, Even though there is a folder, Keep on appending the files
                                       ## inside that whenever we want to create the file.

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

## If we want to overide this functionality of logging, We have to set this up in the Basic Config 

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format = "[%(asctime)s ] %(lineno)d %(name)s -  %(levelname)s: %(message)s",
    level = logging.INFO,

) 

# if __name__=="__main__":
#     logging.info("Logging has started")


## Whenever We use ---> import logging, logging.info , If I write any print message, it is going to use this kind of basic config
## and it is going to create this file path. It is going to keep this particular format with respect to the message and all that
## we are going to get.   


## Logging - Any Execution that happens, we should be able to log all those information, the execution, everything in some files
## so that we will be able to track, if there is some errors, even the Custom Exception Error, Any Exception that comes we will be 
## able to try to log that into the text file. 