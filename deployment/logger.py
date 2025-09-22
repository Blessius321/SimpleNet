import logging
import sys


# Set format
format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Create and configure logger
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                      logging.StreamHandler()

                    ])

# Creating an object
logger = logging.getLogger('root')

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

# # Add handler to display to stdout
# streamHandler = logging.StreamHandler(sys.stdout)
# streamHandler.setFormatter(format)
# logger.addHandler(streamHandler)

