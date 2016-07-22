import os
import argparse
import csv
from   six.moves import cPickle as pkl

def make_sicklike_data(filename, data):
    """
    Make a sick like data for testing
    input:
        filename    (string)    The absolute path for the file we want to create.
        data        (list)      The list contains tuples of two sentences
    """
    with open(filename, 'wb') as csvfile:
        fieldnames = ['pair_ID', 'sentence_A', 'sentence_B',
                'relatedness_score', 'entailment_judgment']
        wr  = csv.DictWriter(csvfile, fieldnames, delimiter='\t')
        idx = 1
        wr.writeheader()
        for sent1, sent2 in data:
            print "Wring the %d-th data"%idx
            wr.writerow({
                "pair_ID"             : idx,
                "sentence_A"          : sent1,
                "sentence_B"          : sent2,
                "relatedness_score"   : 0,
                'entailment_judgment' : 'NEUTRAL'
            })
            idx += 1
    print "Created sick-like annotated file:%s"%filename
    return filename

def make_data_from_logfile(logfile, dformat = 'sick', outfile=None):
    outflie = logfile.split(os.sep)[-1].split('.', 1)[0] if outfile is None else outfile
    logs = pkl.load(open(logfile))
    data = [(x['ablt'], x['orig']) for _, x in logs.iteritems() if isinstance(x, dict)]
    if dformat == 'sick':
        return make_sicklike_data("sick_%s.csv"%outfile, data)
    else:
        print "Can't find the data format pecified: %s"%dformat
    return None

def reattach_scores_to_data(datafile, scorefile, fieldname=None):
    fieldname = "lstm_score" if fieldname is None else fieldname
    print("The fieldname is : %s"%fieldname)
    logs   = pkl.load(open(datafile))
    imgids = [ k for k, v in logs.iteritems() if isinstance(v, dict) ]
    scores = [ float(x) for x in open(scorefile) ]
    dscore = zip(imgids, scores)
    for img_id, s in dscore:
        print("Image id: %s, score : %f"%(str(img_id), s))
        logs[img_id].update({fieldname : s})
    pkl.dump(logs, open(datafile, 'w+'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=['reattach'])
    parser.add_argument('--logfile')
    parser.add_argument('--scorefile')
    parser.add_argument('--fieldname')
    args = parser.parse_args()
    if args.action == 'reattach' :
        reattach_scores_to_data(args.logfile, args.scorefile, args.fieldname)
    else:
        pass
