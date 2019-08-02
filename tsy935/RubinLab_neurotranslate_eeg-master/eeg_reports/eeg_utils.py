import spacy
spacy_en = spacy.load('en_core_web_sm')
from collections import Counter

import re
import pickle

from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar
import numpy as np
from metal.utils import convert_labels

from scipy.sparse import csr_matrix

###################################################################
###################### UTILITY FUNCTIONS ##########################
###################################################################

def create_label_matrix(lfs, docs):
    
    delayed_lf_rows = []
    
    for lf in lfs:
        delayed_lf_rows.append(dask.delayed(evaluate_lf_on_docs)(docs, lf))

    with ProgressBar():
        L = csr_matrix(np.vstack(dask.compute(*delayed_lf_rows)).transpose())  
    
    return L

def evaluate_lf_on_docs(docs, lf):
    """
    Evaluates lf on list of documents
    """
    
    lf_list = []
    for doc in docs:
        lf_list.append(lf(doc))
    return lf_list

def get_section_with_name(section_names, doc):
    """
    Gets text from all sections in section_names from EEGNote doc with 
    """
    text = ''
    for section in section_names:
        try: 
            text = ' '.join([text, doc.sections[section]['text']])
        except:
            pass
        
        try:
            text = ' '.join([text, doc.sections['narrative'][section]])
        except:
            pass
        
        try:
            text = ' '.join([text, doc.sections['findings'][section]])
        except:
            pass
        
    return ' '.join(text.split())

# Utility for getting strings between
def get_string_between(string, val1, val2, inc1=False, inc2=False):
    """
    Get string between two others
    
    PARAMS
    str string: string to search within
    str val1: beginning value
    str val2: end value
    bool inc1: include beginning value in returned string
    bool inc2: include end value in returned string
    
    OUT
    str btw: string between val and val2
    """
    try:
        btw = re.search(f'{val1}(.*?){val2}', string, re.IGNORECASE).group(1)
    except:
        return ''
    if inc1:
        btw = val1+btw
    if inc2:
        btw = btw+val2
    return btw

def check_sec_names(text, sec_names, verbose=False):
    """
    Make sure section names actually exist
    """
    sec_names_out = []
    for name in sec_names:
        try: 
            if re.search(name, text, re.IGNORECASE):
                sec_names_out.append(name)
        except:
            if verbose:
                print('Failed section name search!')
    return sec_names_out

def extract_major_note_sections(text, sec_names):
    """
    Get major note sections
    """
    sec_dict = {}
    sec_names = check_sec_names(text, sec_names)
    for ii, name in enumerate(sec_names):
        if ii < len(sec_names)-1:
            sec_string = get_string_between(text, sec_names[ii], 
                                                sec_names[ii+1], inc1=False, inc2=False)
        else:
            indices = re.finditer(name, text, re.IGNORECASE)
            for idx in indices:
                continue
            sec_string = text[idx.start():]
            sec_string = sec_string[len(name):]
        
        sec_string = sec_string.strip(':').strip()
        #sec_string = sec_string.split('  ')[0] # usually texts between sections are separated by two white spaces
        sec_dict[name] = sec_string
            
    return sec_dict

def extract_general_note_sections(text, sec_names):
    """
    Get general note sections
    """
    sec_dict = {}
    sec_names = check_sec_names(text, sec_names)
    for ii, name in enumerate(sec_names):
        if ii < len(sec_names)-1:
            sec_string = get_string_between(text, sec_names[ii], 
                                                sec_names[ii+1], inc1=False, inc2=False)
        else:
            indices = re.finditer(name, text, re.IGNORECASE)
            for idx in indices:
                continue
            sec_string = text[idx.start():]
            sec_string = sec_string[len(name):]     
        
        sec_string = sec_string.strip(':').strip()
              
        sec_dict[name] = sec_string
            
    return sec_dict

def get_empty_docs(docs):
    """
    Getting EEGnote documents with no sections
    """
    empty_docs = []
    for doc in docs:
        if doc.sections == {}:
            empty_docs.append(doc)
    return empty_docs

def snorkel_to_metal_gold(gold_label):
    if gold_label == 1:
        return 1
    elif gold_label == -1:
        return 2
    elif gold_label is None:
        return None
    
def parse_eeg_docs(df_eeg, use_dask=False):
    """
    Parses EEG documents into EEGNote from dataframe, assumes a
    specific column structure
    
    TODO: Generalized this
    """
    delayed_docs = []
    for i, (_,row) in tqdm(enumerate(df_eeg.iterrows()), total=len(df_eeg)):
        try:
            gold_label = row['hand_label'] if np.isnan(row['hand_label']) != 1 else None
            # Adjusting Snorkel -> Metal Labels
            gold_label = snorkel_to_metal_gold(gold_label)
            if use_dask:
                noteObj = dask.delayed(EEGNote)(row['note_uuid'], row['note'], gold_label=gold_label)
            else:
                noteObj = EEGNote(row['note_uuid'], row['note'], gold_label=gold_label)
            delayed_docs.append(noteObj)
        except:
            print("Bad doc!")
    if use_dask:
        with ProgressBar():
            docs = dask.compute(*delayed_docs)
    else:
        docs = delayed_docs  
    return docs

############# Added by Siyi ############
def parse_eeg_docs_multiclass(df_eeg, column_names, use_dask=False):
    # TODO: Update this function to read 
    """
    Parses EEG documents into EEGNote from dataframe, assumes a
    specific column structure.
    Now, gold_label is a dictionary of different classes
    Args:
        df_eeg: dataframe of EEG file
        column_names: list of column names to extract gold labels
        use_dask: whether to use dask or self-defined EEGNote class   
    """
    delayed_docs = []
    
    for i, (_,row) in tqdm(enumerate(df_eeg.iterrows()), total=len(df_eeg)):
        gold_label = {}
        try:
            for col_name in column_names:
                gold_label[col_name] = row[col_name] if np.isnan(row[col_name]) != 1 else None
                
            if use_dask:
                noteObj = dask.delayed(EEGNote)(row['note_uuid'], row['note'], gold_label=gold_label)
            else:
                noteObj = EEGNote(row['note_uuid'], row['note'], gold_label=gold_label)
            delayed_docs.append(noteObj)
        except:
            print("Bad doc!")
    if use_dask:
        with ProgressBar():
            docs = dask.compute(*delayed_docs)
    else:
        docs = delayed_docs  
    return docs



def create_data_split(docs):

    train_docs, dev_docs, test_docs = [], [], []
    for doc in docs:
        if doc.gold_label is not None:
            if len(dev_docs)>=len(test_docs):
                test_docs.append(doc)
            else:
                dev_docs.append(doc)
        else:
            train_docs.append(doc)
        
    return train_docs, dev_docs, test_docs

###### Added by Siyi ######
def create_data_split_multiclass(docs, keys):

    train_docs, dev_docs, test_docs = [], [], []
    for doc in docs:
        if np.all([doc.gold_label[k] is not None for k in keys]):
            if len(dev_docs)>=len(test_docs):
                test_docs.append(doc)
            else:
                dev_docs.append(doc)
        else:
            train_docs.append(doc)
        
    return train_docs, dev_docs, test_docs


def pickle_model(model, filename):
    with open(filename,'wb') as af:
        pickle.dump(model, af)
        
def unpickle_model(filename):
    with open(filename,'rb') as af:
        model = pickle.load(af)
    return model

###################################################################
###################### EEG SPECIFIC CLASSES #######################
###################################################################

class EEGNote(object):
    def __init__(self, doc_id, text, gold_label=None):
        self.doc_id = doc_id
        self.text = text
        self.tokens = self.tokenize(text)
        self.n = len(self.tokens)
        self.word_counts = Counter(self.tokens)
        self.major_section_names = ['NARRATIVE','NUMBER','DETAILED FINDINGS',
                                    'INTERPRETATION','COMMENTS','FINDINGS','SUMMARY',]
        self.all_section_names = ['NARRATIVE','NUMBER','DETAILED FINDINGS',
                                   'INTERPRETATION','COMMENTS','FINDINGS','SUMMARY',
                                   'eeg date','study dates','test dates','date/time','start date',
                                   'gender','sex','age','date of birth','dob',
                                   'mrn',
                                   'eeg number','eeg#','previous eeg(s)','eeg type',
                                   'loc','location',                                 
                                   'medications','meds','prescriptions','medication',
                                   'requesting md','referring md','referring dept or md','referring dept / md',
                                   'sedation',
                                   'indication for study',
                                   'history','neurology history &amp; indication','brief history &amp; indication for study',
                                   'active prescription orders',
                                   'conditions of recording','condition of recording','alertness',
                                   'background','focal slowing','photic stimulation',
                                   'hyperventilation','epileptiform activity',
                                   'seizures / patient events','seizures/events', 'seizures:', 'seizure:',
                                   'push button activations',
                                   'report prepared by','authorized by','performed by',
                                   'technical notes','duration of study',
                                   'interpreting attending','interpreted by',                                    
                                   'neurobehavioral events','indication',
                                   'impression']

        self.sections = self.create_sections_dict() 
        self.gold_label = gold_label
        
        
    def tokenize(self, txt):
        """
        Using spacy as default tokenizer
        """
        return [tok.text for tok in spacy_en.tokenizer(txt) if (not tok.is_space)]
    
    def count(self,txt_list):
        return Counter(txt_list)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return f"""Document({self.doc_id}, "{self.text[:30]}...")"""
    
    def get_lower_key(self, ky):
        return ' '.join(ky.lower().split())
    
    def get_section_names(self, text):
        note_section_names = []
        str_indices = []
        for name in self.all_section_names:
            str_idx = re.search(name, text, re.IGNORECASE)
            #print(str_idx)
            if str_idx is not None and text[str_idx.start()].isupper():
                sec_str = text[str_idx.start():str_idx.end()]
                note_section_names.append(sec_str)
                str_indices.append(str_idx.start())
        
        # sort section names based on order of occurance in the text
        note_section_names = [x for _,x in sorted(zip(str_indices,note_section_names))]
        return note_section_names 
    
    def create_sections_dict(self):
        sections= {}
        
        # Getting major note sections and updating
        major_note_section_names = []
        general_note_section_names = []
        #note_all_section_names = []
        note_section_names = self.get_section_names(self.text)
        for name in note_section_names:
            if name.upper() in self.major_section_names:
                if name.upper() not in major_note_section_names:
                    major_note_section_names.append(name)
            else:
                if name.lower() not in general_note_section_names:
                    general_note_section_names.append(name)
            
        
        
        major_note_sections = extract_major_note_sections(self.text, major_note_section_names)
        for ky in major_note_sections.keys():
            sections[self.get_lower_key(ky)] = {}
            sections[self.get_lower_key(ky)]['text'] = major_note_sections[ky]
        
        # Getting major section list
        major_note_keys = list(sections.keys())
        #print(major_note_keys)
       
        
        # Getting narrative sections if they exist
        narrative_key = [a for a in ['number', 'report', 'narrative'] if a in major_note_keys]
        
        if narrative_key != []:
            narrative_key = narrative_key[0]
            sections['narrative'] = sections[narrative_key].copy()
            if narrative_key != 'narrative':
                del sections[narrative_key]
            narrative_note_section_names = []
            narrative_section_names = self.get_section_names(sections['narrative']['text'])
            narrative_note_sections = extract_major_note_sections(
            sections['narrative']['text'], narrative_section_names)
            for ky in narrative_note_sections.keys():
                sections['narrative'][self.get_lower_key(ky)] = narrative_note_sections[ky]
                

        # Getting narrative sections if they exist
        findings_key = [a for a in ['findings'] if a in major_note_keys]
        
        if findings_key != []:
            findings_key = findings_key[0]
            sections['findings'] = sections[findings_key]
            if findings_key != 'findings':
                del sections[findings_key]
            findings_note_section_names = []
            findings_section_names = self.get_section_names(sections['findings']['text'])
            findings_note_sections = extract_major_note_sections(
            sections['findings']['text'], findings_section_names)
            for ky in findings_note_sections.keys():
                sections['findings'][self.get_lower_key(ky)] = findings_note_sections[ky]
                                                                 
        # Getting interpretation sections if they exist
        findings_key = [a for a in ['interpretation'] if a in major_note_keys]
        
        if findings_key != []:
            findings_key = findings_key[0]
            sections['interpretation'] = sections[findings_key]
            if findings_key != 'interpretation':
                del sections[findings_key]
            findings_note_section_names = []
            findings_section_names = self.get_section_names(sections['interpretation']['text'])
            findings_note_sections = extract_major_note_sections(
            sections['interpretation']['text'], findings_section_names)
            for ky in findings_note_sections.keys():
                sections['interpretation'][self.get_lower_key(ky)] = findings_note_sections[ky]
                
        # Getting general sections
        general_notes = extract_general_note_sections(self.text, general_note_section_names)     
        #print('general notes keys:{}'.format(general_notes.keys()))
        #print('general notes:{}'.format(general_notes))
        
        for ky in general_notes.keys():
            if ky in sections.keys(): # If there is already a section called the same name, append the text
                sections[self.get_lower_key(ky)] = sections[ky] + general_notes[ky]
            else:
                sections[self.get_lower_key(ky)] = general_notes[ky]

            
        return sections