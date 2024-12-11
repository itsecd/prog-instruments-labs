import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lab_4\\info.log'),
    ]
)


logger = logging.getLogger('roflan_loger')

