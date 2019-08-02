# Initializing parameter dictionary
config = {}

# Adding split sizes
config['train_set_size'] = 0.8 
config['dev_set_size'] = 0.1
config['test_set_size'] = 0.1
    
# Creating postgres db
config['postgres_location'] = 'postgresql:///'
config['postgres_name'] = 'eeg'

# Parallelism
config['parallelism'] = 8