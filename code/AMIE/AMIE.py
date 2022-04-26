from multiprocessing import Queue, Process, Pool

from collections import defaultdict
from copy import deepcopy
import numpy as np
import time
from timeit import default_timer

start = default_timer()
data_dir = '../../data/AMIE/WN18RR/'
minHC = 0.001
maxLen = 311
minConf = 0.01
epsilon = 1e-5

ruleQ = Queue()
outputQ = Queue()

head = defaultdict(set)
tail = defaultdict(set)
hr2t = defaultdict(set)
tr2h = defaultdict(set)
ht2r = defaultdict(set)
r2ht = defaultdict(set)
head2num = defaultdict(int)
tail2num = defaultdict(int)
asHeadOfRelation = defaultdict(set)
asTailOfRelation = defaultdict(set)

def read2id(dir, REL=None):
    item2id = dict()
    id2item = dict()
    items = []
    with open(dir, 'r') as f:
        for l in f.readlines():
            ll = l.strip().split('\t')
            assert len(ll) == 2
            item = ll[0]
            id = ll[1]
            item2id[item] = int(id)
            id2item[int(id)] = item
            items.append(int(id))
            if REL is not None:
                item2id[item+'_inv'] = int(id)+REL
                id2item[int(id)+REL] = item+'_inv'
                items.append(int(id)+REL)
    num_item = len(item2id)
    return item2id, id2item, num_item, items

def record_triple(h,r,t):
    head[r].add(h)
    tail[r].add(t)
    hr2t[(h,r)].add(t)
    tr2h[(t,r)].add(h)
    r2ht[r].add((h,t))
    ht2r[(h,t)].add(r)
    head2num[(r,h)] += 1

trp_dir = data_dir + 'train.txt'
ent2id, id2ent, num_ent, ents = read2id(data_dir + 'entity2id.txt')
rel2id, id2rel, num_rel, rels = read2id(data_dir + 'relation2id.txt', REL=11)
print('#ENT:%d  #REL:%d'%(num_ent, num_rel))

line_num = 0
with open(trp_dir) as f:
    for l in f.readlines():
        line_num += 1
        h,r,t = l.strip().split('\t')
        h = ent2id[h]
        r = rel2id[r]
        t = ent2id[t]
        r_inv = r+num_rel
        record_triple(h,r,t)
        record_triple(t, r_inv, h)


for rel in id2rel.keys():
    ruleQ.put([rel, 0])

def closed(rule):
    return rule[-1] == 1

def length(rule):
    return len(rule)-1

def support(rule, PCA=False):
    lenth = length(rule)
    close = closed(rule)
    if lenth == 2:
        r0 = rule[0]
        r1 = rule[1]
        if close:
            sup = len(r2ht[r0].intersection(r2ht[r1]))
        else:
            intersection_e = head[r0].intersection(head[r1])
            sup = 0
            for h in intersection_e:
                num1 = head2num[(r0,h)]
                num2 = head2num[(r1,h)]
                sup += num1*num2
        pca_num = np.sum(np.asarray([(len(hr2t[(h, r0)]) != 0) * head2num[(r1, h)] for h in head[r1]], dtype=np.int64))
    elif close and lenth ==3:
        r0 = rule[0]
        r1 = rule[1]
        r2 = rule[2]
        sup = sum([len(hr2t[(h, r1)].intersection(tr2h[(t,r2)])) for h,t in r2ht[r0]])
        pca_num = 0
        for h,t in r2ht[r1]:
            pca_num += np.sum(np.asarray([len(hr2t[(h, r0)])!=0 for tt in hr2t[(t,r2)]], dtype=np.int64))
    else:
        print(rule, length(rule))
        raise NotImplementedError('rule do not calculate support')
    if PCA:
        return sup, pca_num
    else:
        return sup

def pca_confidence(rule):
    sup, pca_num = support(rule, PCA=True)
    return (sup/pca_num)

def HC(rule):
    sup = support(rule)
    headr = rule[0]
    return sup/(len(r2ht[headr])+epsilon)


def conf_increase(rule):
    if length(rule) == 2:
        return True
    parent = deepcopy(rule)
    del parent[-2]
    if closed(rule): parent[-1] = 0
    parent_conf = pca_confidence(parent)
    child_conf = pca_confidence(rule)
    return child_conf>parent_conf

def acceptedForOutput(rule):
    if not closed(rule):
        return False
    elif pca_confidence(rule) < minConf:
        return False
    elif not conf_increase(rule):
        return False
    else:
        return True


def addingDanglingAtom(rule):
    assert rule[-1] != 1
    if length(rule) == 2:
        return []

    for rel in rels:
        rule_new = deepcopy(rule)
        rule_new.insert(-1, rel)
        if HC(rule_new) >= minHC:
            yield rule_new

def addingClosingAtom(rule):
    assert rule[-1] != 1
    for i, rel in enumerate(rels):
        rule_new = deepcopy(rule)
        rule_new.insert(-2, i)
        rule_new[-1] = 1
        if HC(rule_new) >= minHC:
            yield rule_new

def task(inputQ, outputQ):
    while True:
        if not inputQ.empty():
            rule = inputQ.get()
            if length(rule)!=1 and acceptedForOutput(rule): outputQ.put(rule)
            if length(rule)<maxLen and not closed(rule):
                for rul in addingDanglingAtom(rule):
                    inputQ.put(rul)
                for rul in addingClosingAtom(rule):
                    inputQ.put(rul)
        else:
            print('task done')
            break

def decodeRules():
    num_rules = 0
    print('RULE     SUPP    HC  CONF_pca')
    while not outputQ.empty():
        rul = outputQ.get()
        num_rules += 1
        assert closed(rul)
        rul_rs = deepcopy(rul)
        del rul_rs[-1]
        rule_str = ''
        for i,r in enumerate(rul[:-1]):
            r = id2rel[r]
            rule_str += r
            if i == 0:
                rule_str += ' <== '
            if i < length(rul)-1 and i != 0:
                rule_str += ' & '
        print('%s   %.d %.4f    %.4f'%(rule_str, support(rul),HC(rul), pca_confidence(rul)))
    print('%d rules are learned'%(num_rules))

if __name__ == '__main__':

    num_worker = 10
    for i in range(num_worker):
        p = Process(target=task, args=(ruleQ, outputQ))
        print('process ' + str(i) + ' start')
        p.start()

    while True:
        if ruleQ.empty():
            print('Rule mining finished')
            decodeRules()
            end = default_timer()
            print('with in %3f'%(end-start))
            break
        else:
            time.sleep(3)
            continue