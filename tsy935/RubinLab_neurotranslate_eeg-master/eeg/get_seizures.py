import sys
import inspect
import os

# Add parent dir to the sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from constants import *
from data.data_utils import getSeizureTuples
 
def main():
    # for cross-validation
    getSeizureTuples(DATA_DIR, verbose=True, split=None)
    
    # for no cross-validation
    getSeizureTuples(TRAIN_DATA_DIR, verbose=True, split='train')
    getSeizureTuples(TEST_DATA_DIR, verbose=True, split='test')
            
if __name__ == "__main__":
    main()
