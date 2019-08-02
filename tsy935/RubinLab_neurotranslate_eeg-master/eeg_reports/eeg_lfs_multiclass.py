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


###### Constant values for labels predicted by each LF ######
# 0 is reserved for abstain
ABSTAIN_VAL = 0
ABNORMAL_VAL = 1
OTHERS_VAL = 2 # other abnormalities
#NA_VAL = 3 # no applicable
ret2str = {ABSTAIN_VAL: 'ABSTAIN_VAL', ABNORMAL_VAL:'ABNORMAL_VAL', OTHERS_VAL:'OTHERS_VAL'}


#ABNORMAL_SLOWING_VAL = 4
#ABNORMAL_SPIKES_VAL = 5
#ABNORMAL_SHARPS_VAL = 6
#ABNORMAL_SUPPRESSION_VAL = 7
#ABNORMAL_DISCONT_VAL = 8
#ABNORMAL_HYPSAR_VAL = 9
#SEIZURE_MOTOR_VAL = 10
#SEIZURE_HYPERKINETIC_VAL = 11
#SEIZURE_TONIC_VAL = 12
#SEIZURE_CLONIC_VAL = 13

#ret2str = { ABSTAIN_VAL:'ABSTAIN_VAL', NO_SEIZURE_VAL:'NO_SEIZURE_VAL', SEIZURE_VAL:'SEIZURE_VAL', 
#            ABNORMAL_SLOWING_VAL:'ABNORMAL_SLOWING_VAL',
#            ABNORMAL_SPIKES_VAL:'ABNORMAL_SPIKES_VAL', ABNORMAL_SHARPS_VAL:'ABNORMAL_SHARPS_VAL',
#            ABNORMAL_SUPPRESSION_VAL:'ABNORMAL_SUPPRESSION_VAL', ABNORMAL_DISCONT_VAL:'ABNORMAL_DISCONT_VAL',
#            ABNORMAL_HYPSAR_VAL:'ABNORMAL_HYPSAR_VAL', SEIZURE_MOTOR_VAL: 'SEIZURE_MOTOR_VAL',
#            SEIZURE_HYPERKINETIC_VAL: 'SEIZURE_HYPERKINETIC_VAL', SEIZURE_TONIC_VAL: 'SEIZURE_TONIC',
#            SEIZURE_CLONIC_VAL: 'SEIZURE_CLONIC'}

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
candidate_interps = ['INTERPRETATION', 'Interpretation', 'Summary', 'impression', 'IMPRESSION', 'conclusion', 'conclusions',
                     'Indication', 'SUMMARY']
CANDIDATE_INTERPS_LOWER = list({ss.lower() for ss in candidate_interps})

## Candidate phrases for seizure events
seizure_section_names = ['seizures/events', 'seizures:', 'push button events', 'neurobehavioral events', 'epileptiform activity']
SEIZURE_SECTION_NAMES_LOWER = CANDIDATE_INTERPS_LOWER + list({ss.lower() for ss in seizure_section_names})


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
            interpretation = report.sections[keyinterp]
            if isinstance(interpretation, dict):
                interp_text = interpretation['text']
            else:
                interp_text = interpretation
            
            if SIMPLE_NORMAL_RE.search(interp_text):
                if NORMAL_STUDY_PHRASES.search(interp_text):
                    #return NO_SEIZURE_VAL
                    return OTHERS_VAL
                    #return NA_VAL
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
            #return SEIZURE_VAL
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL
    else:
        #return NO_SEIZURE_VAL
        return OTHERS_VAL
        #return NA_VAL


def lf_abnormal_interp_with_seizure(report):
    """
    obviously if the study is normal it can't have a seizure
    so look for an abnormal in the interpretation and for a seizure synonym

    problem is that sometimes someone may write "no seizures" so this could be wrong
    see below which makes effort to look for negated phrases
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
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
    
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_seizure(interp_text)
    else:
        candtext = get_section_with_name(CANDIDATE_INTERPS_LOWER, report)
        if candtext:
            return abnormal_interp_with_seizure(candtext)
        else:
            return ABSTAIN_VAL

# Phrases indicating no seizure
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
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        if abnormal_interp_test(interp_text):
            if NOSEIZURE_PHRASE_RE.search(interp_text):
                #return NO_SEIZURE_VAL
                return OTHERS_VAL
            else:
                return ABSTAIN_VAL
        else:
            return OTHERS_VAL 
            #return NA_VAL
    else:
        candtext = get_section_with_name(CANDIDATE_INTERPS_LOWER, report) # note this just joins them all together so can cause probs
        # w/ multi-day reports
        if candtext:
            if abnormal_interp_test(candtext):
                if NOSEIZURE_PHRASE_RE.search(candtext):
                    #return NO_SEIZURE_VAL
                    return OTHERS_VAL
                else:
                    return ABSTAIN_VAL
            else:
                # return NO_SEIZURE_VAL # more agressive
                return OTHERS_VAL # less agressive coverage
                #return NA_VAL
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
        #return NO_SEIZURE_VAL
        return OTHERS_VAL
        #return NA_VAL
    
    parsed_interp = spacy_en(interp)
    neg_found = 0
    seizure_found_and_no_neg = 0

    for sent in parsed_interp.sents:
        # so each sentence
        s = str(sent)
        m1 = BASIC_NEGEX_RE.search(s)
        if m1:
            neg_found=1

        m2 = REVERSED_NEGEX_RE.search(s)
        if m2:
            neg_found =2 

        if not neg_found:
            m3 = SEIZURE_SYNONYMS_RE.search(s)
            if m3:
                seizure_found_and_no_neg = 1 # then at least one sentence said there was a seizure

    if neg_found and not seizure_found_and_no_neg:
        return OTHERS_VAL

    elif seizure_found_and_no_neg:
        #return SEIZURE_VAL
        return ABNORMAL_VAL

    return OTHERS_VAL # its abnormal but we never got a metion of a seizure


def lf_abnl_interp_negexsp_seizure(report):
    """check out if the top 'interpretion' section is abnormal and if so, then look for 
    stuff that indicates if there is or is NOT a seizure using my simple negex-like
    algorithm for SEIZURE

    this will check just the toplevel 'interpetation'

    this is hoped to be a bit more robust that pure regex based version

    """

    for topkey in CANDIDATE_INTERPS_LOWER:
        if topkey in report.sections.keys():
            interpretation = report.sections[topkey]
            if isinstance(interpretation, dict):
                interp_text = interpretation['text']
            else:
                interp_text = interpretation
            
            return eval_interp_with_negex(interp_text)

    return ABSTAIN_VAL # if can't find an interpretation then abstain


def lf_findall_interp_negex_seizure(report):
    candtext = get_section_with_name(CANDIDATE_INTERPS_LOWER, report) # note this just joins them all together so can cause probs
        # w/ multi-day reports
    if candtext:
        return eval_interp_with_negex(candtext)
    else:
        return ABSTAIN_VAL



###### Utility functions ######
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
#############################        


# from jared
# Edited by Siyi
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
        
        try:
            text = ' '.join([text, doc.sections[section]]) # some general sections are sections themselves
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
                    return OTHERS_VAL
                    
                elif 'Many' in seizure_text:
                    #return SEIZURE_VAL
                    return ABNORMAL_VAL
                elif len(seizure_text.split()) > 30:
                    #return SEIZURE_VAL
                    return ABNORMAL_VAL
                else:
                    return OTHERS_VAL
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
        return OTHERS_VAL
        #return NA_VAL
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
        #return SEIZURE_VAL
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL
    
    
def lf_extreme_words_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    reg_abnormal = ['excessive','frequent']
    if any([re.search(reg, impression, re.IGNORECASE) for reg in reg_abnormal] ):
        #return SEIZURE_VAL
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL
    
    
############# Below are for multiclass classification of abnormalities ############
############# Added by Siyi #############

SLOWING_SYNONYMS = r'diffuse\sslowing|slowing|slow|slower'
SLOWING_SYNONYMS_RE = re.compile(SLOWING_SYNONYMS, re.IGNORECASE)

SPIKES_SYNONYMS = r'spike|spikes|spike-waves|spike\swaves|polyspike|polyspike-waves|generalized\sspikes|spike-and-wave|spike/polyspike|spikey'
SPIKES_SYNONYMS_RE = re.compile(SPIKES_SYNONYMS, re.IGNORECASE)

SHARPS_SYNONYMS = r'sharp|sharps|sharply|sharp-waves|sharp\swaves|sharp\stransients|sharper'
SHARPS_SYNONYMS_RE = re.compile(SHARPS_SYNONYMS, re.IGNORECASE)

SUPPRESSION_SYNONYMS = r'burst-suppression|burst\ssuppression|suppression\sburst|suppression-burst'
SUPPRESSION_SYNONYMS_RE = re.compile(SUPPRESSION_SYNONYMS, re.IGNORECASE)

DISCONT_SYNONYMS = r'discontinuous|discontinuity'
DISCONT_SYNONYMS_RE = re.compile(DISCONT_SYNONYMS, re.IGNORECASE)

HYPSAR_SYNONYMS = r'hypsarrhythmia|hysparrhythmia|hysarrhythmia' # hysparrhythmia and hysarrhythmia are typos in the data
HYPSAR_SYNONYMS_RE = re.compile(HYPSAR_SYNONYMS, re.IGNORECASE)

## Abnormal vs normal
def abnormal_interp_in_text(interp_text):
    if abnormal_interp_test(interp_text):
        return ABNORMAL_VAL
    else:
        return OTHERS_VAL
    
def lf_abnormal_interp(report):
    """
    Look for an abnormal in the interpretation
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_in_text(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_in_text(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_in_text(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_in_text(report.sections['findings']['impression'])
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_in_text(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_in_text(report.sections[ky]['impression'])
        # could try running on all of findings but would really need to check then
        
        return ABSTAIN_VAL        
    else:
        return ABSTAIN_VAL
    

def lf_findall_abnormal_interp(report):
    """check out if the top 'interpertion' section is abnormal
    """

    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_in_text(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return abnormal_interp_in_text(candtext)
        else:
            return ABSTAIN_VAL
    
def lf_normal_interp(report):
    """
    Getting impression section, checking for regexes
    """
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    reg_normal = ['normal EEG', 'normal aEEG', 'EEG is normal', 'normal study']
    if any([re.search(reg, impression, re.IGNORECASE) for reg in reg_normal] ):
        return OTHERS_VAL
    else:
        return ABSTAIN_VAL


## Slowing
def abnormal_interp_with_slowing(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SLOWING_SYNONYMS_RE.search(interp_text):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL # abnormal but not slowing
    else:
        return OTHERS_VAL # normal
        #return NA_VAL # normal
    
def lf_abnormal_interp_with_slowing(report):
    """
    Look for an abnormal in the interpretation and for a slowing synonym
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_slowing(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_with_slowing(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_with_slowing(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_with_slowing(report.sections['findings']['impression'])
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_with_slowing(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_with_slowing(report.sections[ky]['impression'])
        # could try running on all of findings but would really need to check then
        
        return ABSTAIN_VAL        
    else:
        return ABSTAIN_VAL
    

def lf_findall_interp_with_slowing(report):
    """check out if the top 'interpertion' section is abnormal and if so, then look for 
    stuff that indicates there is slowing.
    uses 'abnormal_interp_with_slowing' function for this.
    """

    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_slowing(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return abnormal_interp_with_slowing(candtext)
        else:
            return ABSTAIN_VAL
        
        
def lf_slowing_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    if SLOWING_SYNONYMS_RE.search(impression):
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL
    
    
## Spikes
def abnormal_interp_with_spikes(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SPIKES_SYNONYMS_RE.search(interp_text):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL # abnormal but not spikes
            
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_abnormal_interp_with_spikes(report):
    """
    Look for an abnormal in the interpretation and for a spikes synonym
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_spikes(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_with_spikes(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_with_spikes(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_with_spikes(report.sections['findings']['impression'])
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_with_spikes(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_with_spikes(report.sections[ky]['impression'])
        # could try running on all of findings but would really need to check then
        
        return ABSTAIN_VAL        
    else:
        return ABSTAIN_VAL
    

def lf_findall_interp_with_spikes(report):
    """check out if the top 'interpertion' section is abnormal and if so, then look for 
    stuff that indicates there is spikes.
    uses 'abnormal_interp_with_spikes' function for this.
    """

    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_spikes(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return abnormal_interp_with_spikes(candtext)
        else:
            return ABSTAIN_VAL

def lf_spikes_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    if SPIKES_SYNONYMS_RE.search(impression):
        #return SEIZURE_VAL
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL

## Sharps
def abnormal_interp_with_sharps(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SHARPS_SYNONYMS_RE.search(interp_text):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL # abnormal but not sharps
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_abnormal_interp_with_sharps(report):
    """
    Look for an abnormal in the interpretation and for a sharps synonym
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_sharps(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_with_sharps(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_with_sharps(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_with_sharps(report.sections['findings']['impression'])
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_with_sharps(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_with_sharps(report.sections[ky]['impression'])        
        return ABSTAIN_VAL        
    else:
        return ABSTAIN_VAL    

def lf_findall_interp_with_sharps(report):
    """check out if the top 'interpertion' section is abnormal and if so, then look for 
    stuff that indicates there is sharps.
    uses 'abnormal_interp_with_sharps' function for this.
    """

    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_sharps(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return abnormal_interp_with_sharps(candtext)
        else:
            return ABSTAIN_VAL
        
def lf_sharps_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    if SHARPS_SYNONYMS_RE.search(impression):
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL 
    
        
## Suppression
def abnormal_interp_with_suppression(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SUPPRESSION_SYNONYMS_RE.search(interp_text):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL # abnormal but not burst suppression
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
    
def lf_abnormal_interp_with_suppression(report):
    """
    Look for an abnormal in the interpretation and for a suppression synonym
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_suppression(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_with_suppression(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_with_suppression(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_with_suppression(report.sections['findings']['impression']) 
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_with_suppression(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_with_suppression(report.sections[ky]['impression'])        
        return ABSTAIN_VAL      
    else:
        return ABSTAIN_VAL    

def lf_findall_interp_with_suppression(report):
    """check out if the top 'interpertion' section is abnormal and if so, then look for 
    stuff that indicates there is suppression.
    uses 'abnormal_interp_with_suppression' function for this.
    """

    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_suppression(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return abnormal_interp_with_suppression(candtext)
        else:
            return ABSTAIN_VAL
        
def lf_suppression_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    if SUPPRESSION_SYNONYMS_RE.search(impression):
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL 
    
    
## Discontinuity
def abnormal_interp_with_discont(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if DISCONT_SYNONYMS_RE.search(interp_text):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL # abnormal but not discontinuity
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_abnormal_interp_with_discont(report):
    """
    Look for an abnormal in the interpretation and for a suppression synonym
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_discont(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_with_discont(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_with_discont(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_with_discont(report.sections['findings']['impression']) 
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_with_discont(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_with_discont(report.sections[ky]['impression'])        
        return ABSTAIN_VAL        
    else:
        return ABSTAIN_VAL    

def lf_findall_interp_with_discont(report):
    """check out if the top 'interpertion' section is abnormal and if so, then look for 
    stuff that indicates there is discontinuity.
    uses 'abnormal_interp_with_discont' function for this.
    """

    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_discont(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return abnormal_interp_with_discont(candtext)
        else:
            return ABSTAIN_VAL

def lf_discont_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    if DISCONT_SYNONYMS_RE.search(impression):
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL
    
## Hysarrhythmia
def abnormal_interp_with_hypsarrhythmia(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if HYPSAR_SYNONYMS_RE.search(interp_text):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL # abnormal but hypsarrhythmia
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_abnormal_interp_with_hypsarrhythmia(report):
    """
    Look for an abnormal in the interpretation and for a hypsarrhythmia synonym
    """
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_hypsarrhythmia(interp_text)
    elif 'summary' in report.sections:
        return abnormal_interp_with_hypsarrhythmia(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return abnormal_interp_with_hypsarrhythmia(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return abnormal_interp_with_hypsarrhythmia(report.sections['findings']['impression'])  
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return abnormal_interp_with_hypsarrhythmia(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return abnormal_interp_with_hypsarrhythmia(report.sections[ky]['impression'])        
        return ABSTAIN_VAL        
    else:
        return ABSTAIN_VAL    

def lf_findall_interp_with_hypsarrhythmia(report):
    """check out if the top 'interpertion' section is abnormal and if so, then look for 
    stuff that indicates there is hypsarrhythmia.
    uses 'abnormal_interp_with_hypsarrhythmia' function for this.
    """

    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return abnormal_interp_with_hypsarrhythmia(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return abnormal_interp_with_hypsarrhythmia(candtext)
        else:
            return ABSTAIN_VAL

def lf_hypsarrhythmia_in_impression(report):
    impression_words = ['impression','interpretation','comments']
    impression = get_section_with_name(impression_words, report)
    if HYPSAR_SYNONYMS_RE.search(impression):
        return ABNORMAL_VAL
    else:
        return ABSTAIN_VAL
    

############# Below are for multiclass classification of seizures ############
############# Added by Siyi #############

## Currently, only implemented classes that have examples in the dataset
"""
SEIZURE_MOTOR_VAL = 10
SEIZURE_HYPERKINETIC_VAL = 11
SEIZURE_TONIC_VAL = 12
SEIZURE_CLONIC_VAL = 13
"""

SEIZURE_MOTOR_SYN = r'tonic|tonically|clonic|clonically|atonic|myoclonic|myoclonia| \
                      spasm|spasms|paralyzed|paralysis|dystonic|dystonia| \
                      automatism|automatims|dysarthria|hypermotor|hyperkinetic| \
                      negative\smyoclonus|version|motor| \
                      fencing\sposture|bilateral\smotor|figure\sof\s4|fencer|figure-four| \
                      epilepsia\spartialis\scontinua'
SEIZURE_MOTOR_RE = re.compile(SEIZURE_MOTOR_SYN, re.IGNORECASE)

NEG_DYSARTHRIA_SYN = r'no\sdysarthria'
NEG_DYSARTHRIA_SYN_RE = re.compile(NEG_DYSARTHRIA_SYN, re.IGNORECASE)

SEIZURE_HYPERKINETIC_SYN = r'hypermotor|hyperkinetic'
SEIZURE_HYPERKINETIC_RE = re.compile(SEIZURE_HYPERKINETIC_SYN, re.IGNORECASE)

SEIZURE_CLONIC_SYN = r'clonic'
SEIZURE_CLONIC_RE = re.compile(SEIZURE_CLONIC_SYN, re.IGNORECASE)

NEG_SEIZURE_CLONIC_SYN = r'no\sclonic'
NEG_SEIZURE_CLONIC_RE = re.compile(NEG_SEIZURE_CLONIC_SYN, re.IGNORECASE)

SEIZURE_TONIC_CLONIC_SYN = r'tonic\sclonic|tonic-clonic'
SEIZURE_TONIC_CLONIC_RE = re.compile(SEIZURE_TONIC_CLONIC_SYN, re.IGNORECASE)

SEIZURE_TONIC_SYN = r'tonic'
SEIZURE_TONIC_RE = re.compile(SEIZURE_TONIC_SYN, re.IGNORECASE)

NEG_SEIZURE_TONIC_SYN = r'no\stonic'
NEG_SEIZURE_TONIC_RE = re.compile(NEG_SEIZURE_TONIC_SYN, re.IGNORECASE)

SEIZURE_MYOCLONIC_SYN = r'myoclonic'
SEIZURE_MYOCLONIC_RE = re.compile(SEIZURE_MYOCLONIC_SYN, re.IGNORECASE)

## Motor seizure
def seizure_motor(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SEIZURE_MOTOR_RE.search(interp_text) \
           and (not NEG_DYSARTHRIA_SYN_RE.search(interp_text)) \
           and (not NEG_SEIZURE_CLONIC_RE.search(interp_text)) \
           and (not NEG_SEIZURE_TONIC_RE.search(interp_text)):
                
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_seizure_motor(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_motor(interp_text)
    elif 'summary' in report.sections:
        return seizure_motor(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return seizure_motor(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return seizure_motor(report.sections['findings']['impression'])      
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return seizure_motor(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return seizure_motor(report.sections[ky]['impression'])        
        return ABSTAIN_VAL
    else:
        return ABSTAIN_VAL    

def lf_findall_seizure_motor(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_motor(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return seizure_motor(candtext)
        else:
            return ABSTAIN_VAL
        
def lf_seizure_motor_in_sections(report):
    sections = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
    if sections:        
        return seizure_motor(sections)
    else:
        return ABSTAIN_VAL
    
## Hyperkinetic seizure
def seizure_hyperkinetic(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SEIZURE_HYPERKINETIC_RE.search(interp_text):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_seizure_hyperkinetic(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_hyperkinetic(interp_text)
    elif 'summary' in report.sections:
        return seizure_hyperkinetic(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return seizure_hyperkinetic(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return seizure_hyperkinetic(report.sections['findings']['impression']) 
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return seizure_hyperkinetic(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return seizure_hyperkinetic(report.sections[ky]['impression'])        
        return ABSTAIN_VAL
    else:
        return ABSTAIN_VAL    

def lf_findall_seizure_hyperkinetic(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_hyperkinetic(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return seizure_hyperkinetic(candtext)
        else:
            return ABSTAIN_VAL
        
def lf_seizure_hyperkinetic_in_sections(report):
    sections = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
    if sections:        
        return seizure_hyperkinetic(sections)
    else:
        return ABSTAIN_VAL
    
        
## Clonic seizure
def seizure_clonic(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SEIZURE_CLONIC_RE.search(interp_text) and (not NEG_SEIZURE_CLONIC_RE.search(interp_text)) \
         and (not SEIZURE_TONIC_CLONIC_RE.search(interp_text)) and (not SEIZURE_MYOCLONIC_RE.search(interp_text)):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_seizure_clonic(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_clonic(interp_text)
    elif 'summary' in report.sections:
        return seizure_clonic(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return seizure_clonic(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return seizure_clonic(report.sections['findings']['impression'])   
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return seizure_clonic(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return seizure_clonic(report.sections[ky]['impression'])        
        return ABSTAIN_VAL
    else:
        return ABSTAIN_VAL    

def lf_findall_seizure_clonic(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_clonic(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return seizure_clonic(candtext)
        else:
            return ABSTAIN_VAL

def lf_seizure_clonic_in_sections(report):
    sections = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
    if sections:        
        return seizure_clonic(sections)
    else:
        return ABSTAIN_VAL
        
## Tonic seizure
def seizure_tonic(interp_text):
    if ABNORMAL_RE.search(interp_text):
        if SEIZURE_TONIC_RE.search(interp_text) and (not NEG_SEIZURE_TONIC_RE.search(interp_text)) \
          and (not SEIZURE_TONIC_CLONIC_RE.search(interp_text)):
            return ABNORMAL_VAL
        else:
            return OTHERS_VAL
    else:
        return OTHERS_VAL # normal
        #return NA_VAL
    
def lf_seizure_tonic(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_tonic(interp_text)
    elif 'summary' in report.sections:
        return seizure_tonic(report.sections['summary']['text'])
    elif 'findings' in report.sections: # fall back to look in the findings 
        if 'summary' in report.sections['findings']: # fall back to look for a summary instead
            return seizure_tonic(report.sections['findings']['summary'])
        if 'impression' in report.sections['findings']:
            return seizure_tonic(report.sections['findings']['impression']) 
        return ABSTAIN_VAL
    elif 'narrative' in report.sections: # fall back to look in the findings 
        ky = 'narrative'
        if 'summary' in report.sections[ky]: # fall back to look for a summary instead
            return seizure_tonic(report.sections[ky]['summary'])
        if 'impression' in report.sections[ky]:
            return seizure_tonic(report.sections[ky]['impression'])        
        return ABSTAIN_VAL
    else:
        return ABSTAIN_VAL    

def lf_findall_seizure_tonic(report):
    if 'interpretation' in report.sections.keys():
        interpretation = report.sections['interpretation']
        interp_text = interpretation['text']
        return seizure_tonic(interp_text)
    else:
        candtext = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
        if candtext:
            return seizure_tonic(candtext)
        else:
            return ABSTAIN_VAL
        
def lf_seizure_tonic_in_sections(report):
    sections = get_section_with_name(SEIZURE_SECTION_NAMES_LOWER, report)
    if sections:        
        return seizure_tonic(sections)
    else:
        return ABSTAIN_VAL
        