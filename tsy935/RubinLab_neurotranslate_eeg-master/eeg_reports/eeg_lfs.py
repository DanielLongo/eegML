import re
from pprint import pformat
import textwrap

import logging
logger = logging.getLogger('testlf')
hdlr = logging.FileHandler('testlf.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.DEBUG)

import spacy

spacy_en = spacy.load('en_core_web_sm')
# spacy_en = spacy.load('en_core_web_lg')

# Writing LFs
# TODO: write code to split multi-day reports into individual day reports
# could use a python enum instead
ABSTAIN_VAL = 0
SEIZURE_VAL = 1
NO_SEIZURE_VAL = 2
# if want to get into things beyond seizures vs not seizures
ABNORMAL_EEG_VAL = 3

ret2str = { ABSTAIN_VAL:'ABSTAIN_VAL', NO_SEIZURE_VAL:'NO_SEIZURE_VAL', SEIZURE_VAL:'SEIZURE_VAL', 
            ABNORMAL_EEG_VAL:'ABNORMAL_EEG_VAL'}

### define useful regular expressions. Can we do better with spacy_en parser?
SIMPLE_NORMAL_RE = re.compile('\snormal\s', re.IGNORECASE)
# note if used spacy lemma then could I avoid having to use (is|was) ?
EEGSYN = r'(EEG|study|record|electroencephalogram|ambulatory\s+EEG|video.EEG\sstudy)'
# can I use spacy to find all entities which have EEG or study in them?
NORMAL_STUDY_PHRASES = re.compile(rf'\snormal\s+{EEGSYN}'
                                  rf'|\snormal\s+awake\s+and\s+asleep\s+{EEGSYN}'
                                  rf'|\snormal\s+awake\s+{EEGSYN}'
                                  rf'|\snormal\s+awake\s+and\s+drowsy\s+{EEGSYN}'
                                  rf'|\snormal\s+asleep\s+{EEGSYN}'
                                  rf'|\s{EEGSYN}\s+(is|was)\s+normal'
                                  rf'|\srange\s+of\s+normal'  # generous
                                  rf'|\s(is|was)\s+normal\s+for\s+age'
                                  #rf'|(EEG|study|record)\s+(is|was)\s+normal\s+for\s+age'
                                  #rf'|(EEG|study|record)\s+(is|was)\s+normal\s+for\s+age'                                  
                                  rf'|{EEGSYN}\s+(is|was)\s+within\s+normal\s+'
                                  rf'|{EEGSYN}\s+(is|was)\s+borderline\+snormal'
                                  rf'|{EEGSYN}\s+(is|was)\s+at\s+the\s+borderline\s+of\s+being\s+normal'
                                  rf'|{EEGSYN}\s+capturing\s+wakefulness\s+and\s+sleep\s+(is|was)\s+normal'
                                  rf'|{EEGSYN}\s+capturing\s+wakefulness\s+(is|was)\s+normal',
                                  re.IGNORECASE)
# This EEG is normal <- if could reduce a sentence to this in spacy would be a more general response
# ? "
# This is a normal EEG
# This normal EEG
# This EEG in the awake state only is within normal limits for age, without evidence of focality or epileptiform activity.
# This EEG is at the borderline of being normal for age, with some potential activity indicating intermittent focal slowing bilateral temporal regions, though this activity may be the result of artifact.
# This EEG capturing  is normal for age, without evidence of focality or epileptiform activity.
# This EEG capturing wakefulness and drowsiness is normal for age, without evidence of focality
# This EEG in the awake stage only is normal for age, without evidence of focality or epileptiform activity.

ABNORMAL_RE = re.compile(r'abnormal', re.IGNORECASE)
# This EEG is abnormal
# Abnormal EEG
# Abnormal continuous EEG
# This record is probably abnormal in
# This is an abnormal EEG

SEIZURE_SYNONYMS = r'seizure|spasm|seizures|status\sepilepticus|epilepsia\spartialis\scontinua|drop\sattack' # ?'myoclonic jerk'
SEIZURE_SYNONYMS_RE = re.compile(SEIZURE_SYNONYMS, re.IGNORECASE)

NEG_DET = ['no', 'not', 'without'] # 'denies'

NEG_SEIZURE = r'no seizures|no epileptiform activity or seizures'.replace(' ','\s')  # does not account for things like no spikes or seizures can spacy parse this?
NEG_SEIZURE_RE = re.compile(NEG_SEIZURE, re.IGNORECASE)


# section keys which we might want which seem like INTERPRETATION sections 
candidate_interps = ['INTERPRETATION', 'Interpretation', 'Summary', 'impression', 'IMPRESSION', 'conclusion', 'conclusions']
CANDIDATE_INTERPS_LOWER = list({ss.lower() for ss in candidate_interps})


def is_not_abnormal_interp(interp):
    """
    check text of interpretation to see if we think it is normal
    this is very primitive to start with but it might be good enough
    """
    m = ABNORMAL_RE.search(interp)
    if not m:
        return True
    else:
        return False



def lf_normal_interp_not_seizure(report):
    """
    obviously if the study is normal it can't have a seizure
    this labeling funciton just looks for an top level interpretation section 
    so it it very safe at giving NO_SEIZURE_VAL, but misses a bunch
    """
    # print(report)

    for keyinterp in CANDIDATE_INTERPS_LOWER:
        if keyinterp in report.sections.keys():
            #print(report.sections.keys())
            interpretation = report.sections[keyinterp]
            #print('   ', interpretation)
            interp_text = interpretation['text']
            
            #print(type(interp_text), interp_text)
            if SIMPLE_NORMAL_RE.search(interp_text):
                # print(f'quick check: {interp_text}')
                if NORMAL_STUDY_PHRASES.search(interp_text):
                    return NO_SEIZURE_VAL
                else:
                    logger.info(f'warning did not get second normal match: {interp_text}')
                    return ABSTAIN_VAL
                
            else:
                return ABSTAIN_VAL

    return ABSTAIN_VAL

def abnormal_interp_test(interp_text):
    return ABNORMAL_RE.search(interp_text)    

def abnormal_interp_with_seizure(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SEIZURE_SYNONYMS_RE.search(interp_text):
            return SEIZURE_VAL
        else:
            return NO_SEIZURE_VAL
    else:
        return NO_SEIZURE_VAL


def lf_abnormal_interp_with_seizure(report):
    """
    obviously if the study is normal it can't have a seizure
    so look for an abnormal in the interpretation and for a seizure synonym

    problem is that sometimes someone may write "no seizures" so this could be wrong
    see below which makes effort to look for negated phrases
    """
    # print(report)
    if 'interpretation' in report.sections.keys():
        #print(report.sections.keys())
        interpretation = report.sections['interpretation']
        #print('   ', interpretation)
        interp_text = interpretation['text']
        #print(type(interp_text), interp_text)
        return abnormal_interp_with_seizure(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_with_seizure(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_with_seizure(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_with_seizure(report.sections['findings']['impression'])
        # could try running on all of findings but would really need to check then
        
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_with_seizure(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_with_seizure(report.sections[ky]['impression'])
        # could try running on all of findings but would really need to check then
        
        return ABSTAIN_VAL

        
    else:
        return ABSTAIN_VAL



def lf_findall_interp_with_seizure(report):
    """check out if the top 'interpertion' section is abnormal and if so, then look for 
    stuff that indicates there is a seizure.
    uses 'abnormal_interp_with_seizure' function for this.

    then if there was not toplevel 'interpetation', this looks to find all the
    things that are broken out into interp-like sections no matter where in the
    hierachy they show up obviously if the study is normal it can't have a
    seizure so look for an abnormal in the interpretation and for a seizure
    synonym

    expect this to abstain when study is abnormal but it does not have seizures mentioned in the interp
    problem: sometimes someone may write "no seizures"

    """
    # print(report)

    if 'interpretation' in report.sections.keys():
        #print(report.sections.keys())
        interpretation = report.sections['interpretation']
        #print('   ', interpretation)
        interp_text = interpretation['text']
        #print(type(interp_text), interp_text)
        return abnormal_interp_with_seizure(interp_text)
    else:
        candtext = get_section_with_name(CANDIDATE_INTERPS_LOWER, report)
        if candtext:
            return abnormal_interp_with_seizure(candtext)
        else:
            return ABSTAIN_VAL

NOSEIZURE_PHRASE_RE = re.compile(r'\bno seizures\b|\bno\sepileptiform\sactivity\sor\sseizures\b'
                      r'|\bno findings to indicate seizures\b'
                      r'|no findings to indicate'
                      r'|no new seizures'
                      r'|with no seizures'
                      r'|no evidence to support seizures'
                      r'|nonepileptic'
                      r'|non-epileptic'
                      #r'|absence of epileptiform'
                      ,                      
                      re.IGNORECASE|re.UNICODE)

def lf_findall_abnl_interp_without_seizure(report):
    """check out if the top 'interpretion' section is abnormal and if so, then look for 
    stuff that indicates there is NOT a seizure.

    then if there was not toplevel 'interpetation', this looks to find all the
    things that are broken out into interp-like sections no matter where in the
    hierachy they show up obviously if the study is normal it can't have a
    seizure so look for an abnormal in the interpretation and for a seizure
    synonym

    expect this to abstain when study is abnormal but it does not have seizures mentioned in the interp
    problem: sometimes someone may write "no seizures"

    """
    # print(report)

    if 'interpretation' in report.sections.keys():
        #print(report.sections.keys())
        interpretation = report.sections['interpretation']
        #print('   ', interpretation)
        interp_text = interpretation['text']
        #print(type(interp_text), interp_text)
        if abnormal_interp_test(interp_text):
            if NOSEIZURE_PHRASE_RE.search(interp_text):
                return NO_SEIZURE_VAL
            else:
                return ABSTAIN_VAL
        else:
            return ABSTAIN_VAL 
    else:
        candtext = get_section_with_name(CANDIDATE_INTERPS_LOWER, report) # note this just joins them all together so can cause probs
        # w/ multi-day reports
        if candtext:
            if abnormal_interp_test(candtext):
                if NOSEIZURE_PHRASE_RE.search(candtext):
                    return NO_SEIZURE_VAL
                else:
                    return ABSTAIN_VAL
            else:
                # return NO_SEIZURE_VAL # more agressive
                return ABSTAIN_VAL # less agressive coverage

        else:
            return ABSTAIN_VAL

###### NEGEX WORK ######
# this was prototyped in develop_neg_labeling.org
NEG_DET= r'(\bno\b|\bnot\b|\bwithout\sfurther\b|\bno\sfurther\b|without|neither)'
# r'|negative for seizures'
# r'|no evidence for seizures'
SEIZURE_SYNONYMS = r'seizure|seizures|spasm|spasms|status\sepilepticus|epilepsia\spartialis\scontinua|drop\sattack'
SEIZURE_SYNONYMS_RE = re.compile(SEIZURE_SYNONYMS, re.IGNORECASE|re.UNICODE)

# <SEIZURE> * <NEG> or <NEG> * <SEIZURE>
BASIC_NEGEX_RE = re.compile(NEG_DET + '.*('+ SEIZURE_SYNONYMS + ')', re.IGNORECASE|re.UNICODE) # meant for single sentence application
REVERSED_NEGEX_RE = re.compile('('+ SEIZURE_SYNONYMS + ').*' + NEG_DET, re.IGNORECASE|re.UNICODE)

# errors: using whole sentence as scope has some problems -- consider [This record is abnormal due to focal status epilepticus affecting the right hemisphere, and moderate diffuse slowing across the left hemisphere without involvement of seizure activity.,]

def eval_interp_with_negex(interp):
    """
    use a NegEx like algorithm, 
    restrict scope to spacy sentence instead of using
    5 token scope used in NegEx paper

    looks at each sentence if a sentence says there is a seizure
    then that overrides all the negative sentences
    # output
    :   Polarity  Coverage  Overlaps  Conflicts  Correct  Incorrect  Emp. Acc.
    : 0   [1, 2]  0.629371       0.0        0.0       88          2   0.977778

    """
    # if we know it is normal, then we don't need to do much
    if is_not_abnormal_interp(interp):
        return NO_SEIZURE_VAL
    
    parsed_interp = spacy_en(interp)
    neg_found = 0
    seizure_found_and_no_neg = 0

    for sent in parsed_interp.sents:
        # so each sentence
        s = str(sent)
        m1 = BASIC_NEGEX_RE.search(s)
        if m1:
            neg_found=1
            # print(m1)

        m2 = REVERSED_NEGEX_RE.search(s)
        if m2:
            neg_found =2 
            # print(neg_found, m2)

        if not neg_found:
            m3 = SEIZURE_SYNONYMS_RE.search(s)
            if m3:
                seizure_found_and_no_neg = 1 # then at least one sentence said there was a seizure

    if neg_found and not seizure_found_and_no_neg:
        return NO_SEIZURE_VAL

    elif seizure_found_and_no_neg:
        return SEIZURE_VAL

    return NO_SEIZURE_VAL # its abnormal but we never got a metion of a seizure so we guess, not seizures 


def lf_abnl_interp_negexsp_seizure(report):
    """check out if the top 'interpretion' section is abnormal and if so, then look for 
    stuff that indicates if there is or is NOT a seizure using my simple negex-like
    algorithm for SEIZURE

    this will check just the toplevel 'interpetation'

    this is hoped to be a bit more robust that pure regex based version

    """

    for topkey in CANDIDATE_INTERPS_LOWER:
        if topkey in report.sections.keys():
            #print(report.sections.keys())
            interpretation = report.sections[topkey]
            #print('   ', interpretation)
            interp_text = interpretation['text']
            #print(type(interp_text), interp_text)
            return eval_interp_with_negex(interp_text)

    return ABSTAIN_VAL # if can't find an interpretation then abstain

def lf_findall_interp_negex_seizure(report):
    candtext = get_section_with_name(CANDIDATE_INTERPS_LOWER, report) # note this just joins them all together so can cause probs
        # w/ multi-day reports
    if candtext:
        return eval_interp_with_negex(candtext)
    else:
        return ABSTAIN_VAL




def flatten_sections_structure(report):
    """first do  depth first walk
    then return a list of (key,val) tuples - like OrderedDict
    """
    allitems = []

    def walk_dict(d):
        if isinstance(d, dict):
            for kk in d:
                if isinstance(d[kk], dict):
                    walk_dict(d[kk])
                else:
                    allitems.append((kk, d[kk]))
        else:
            raise Exception('did not handle this case')
        return
    walk_dict(report.sections)

    return allitems


def flatten_sections_filter_keys(report, key_filter=None):
    """first do  depth first walk
    then return a list of (key,val) tuples - like OrderedDict
    where key_filter(key) == True or truthy
    """
    allitems = []

    if not key_filter:
        def walk_dict(d):
            if isinstance(d, dict):
                for kk in d:
                    if isinstance(d[kk], dict):
                        walk_dict(d[kk])
                    else:
                        allitems.append((kk, d[kk]))
            else:
                raise Exception('did not handle this case')
            return
        walk_dict(report.sections)

        return allitems
    else:
        def walk_dict(d):
            if isinstance(d, dict):
                for kk in d:
                    if isinstance(d[kk], dict):
                        walk_dict(d[kk])
                    else:
                        if key_filter(kk):
                            allitems.append((kk, d[kk]))
            else:
                raise Exception('did not handle this case')
            return
        walk_dict(report.sections)

        return allitems
        


# from jared
def get_section_with_name(section_names, doc):
    """
    check exact matches for keys in section_names
    this presumes a certain structure in EEGNote
    (was written by Jared)
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
            
def evaluate_lf_on_docs(docs, lf):
    lf_list = []
    for doc in docs:
        lf_list.append(lf(doc))
    return lf_list



def lf_seizure_section(report):
    """
    Checking to see if there is a "seizure" section in the report and if it is indicative one way or the other
    
    """
    if 'findings' in report.sections.keys():
        seizure_keys = [key for key in report.sections['findings'].keys() if 'seizure' in key ]
        if not seizure_keys:
            return ABSTAIN_VAL
        else:
            for ky in seizure_keys:
                seizure_text = report.sections['findings'][ky]
                if 'None' in seizure_text:
                    return NO_SEIZURE_VAL
                elif 'Many' in seizure_text:
                    return SEIZURE_VAL
                elif len(seizure_text.split()) > 30:
                    return SEIZURE_VAL
                else:
                    return NO_SEIZURE_VAL
    else:
        return ABSTAIN_VAL
    
def lf_impression_section_negative(report):
    """
    Getting impression section, checking for regexes
    """
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    reg_normal = ['no epileptiform', 'absence of epileptiform', 'not epileptiform', 
                  'normal EEG', 'normal aEEG','benign','non-specific','nonepileptic','idiopathic',
                  'no seizures','EEG is normal','normal study']
    if any([re.search(reg, impression, re.IGNORECASE) for reg in reg_normal] ):
        return NO_SEIZURE_VAL
    else:
        return ABSTAIN_VAL
    
def lf_impression_section_positive(report):
    """
    Getting impression section, checking for regexes
    """
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)

    reg_abnormal = ['status epilepticus','spasms','abnormal continuous',
                    'tonic','subclinical','spike-wave', 'markedly abnormal']  
    if any([re.search(reg, impression, re.IGNORECASE) for reg in reg_abnormal] ):
        return SEIZURE_VAL
    else:
        return ABSTAIN_VAL
    
    
def lf_spikes_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    if re.search('spike',impression,re.IGNORECASE):
        return SEIZURE_VAL
    else:
        return ABSTAIN_VAL
    
def lf_extreme_words_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    reg_abnormal = ['excessive','frequent']
    if any([re.search(reg, impression, re.IGNORECASE) for reg in reg_abnormal] ):
        return SEIZURE_VAL
    else:
        return ABSTAIN_VAL
