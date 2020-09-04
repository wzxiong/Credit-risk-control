import sys
import math
from collections import namedtuple # can be accessed by either name or index

dev_fn, oot_fn = sys.argv[1:3]

# assume the schema of each file is :
# bid, adate, fico_new, fico_ref, lbl

def read_set(fn):
    FICOs = namedtuple('FICOs', ['fico'])
    contents = []
    with open(fn) as f:
        for line in f:
        #fico = line.rstrip('\n').split(',')
            fico = line.rstrip('\n').split(',')
            try:
                fico = FICOs(fico=float(fico[0]))
            except:
                sys.stderr.write(line)
                continue
            contents.append(fico)
    return contents

def cal_psi(dev, oot, score_idx, buckets=10):
    dev.sort(key=lambda x: x[score_idx])
    total_cnt = len(dev)
    buk_len = int(float(total_cnt)/buckets)
    buk_info = [{'l_b':9999,
                 'u_b':-1,
                 's_idx':-1,
                 'e_idx':-1,
                 'total_cnt':0.,
                 'ratio':0.,
                 'actual_total':0.,
                 'actual_ratio':0.}
                 for i in range(buckets)]
    buk_info[0]['l_b'] = -1
    buk_info[buckets-1]['u_b'] = 9999
    total_offset = 0
    for i in range(buckets):
        if i*buk_len+total_offset >= len(dev):
            break
        buk_info[i]['s_idx'] = i*buk_len+total_offset
        for j in range(i*buk_len+total_offset, (i+1)*buk_len+total_offset):
            if j >= len(dev):
                break

            if buk_info[i]['l_b'] > dev[j][score_idx]:
                 buk_info[i]['l_b'] = dev[j][score_idx]
            if buk_info[i]['u_b'] < dev[j][score_idx]:
                 buk_info[i]['u_b'] = dev[j][score_idx]
            buk_info[i]['total_cnt'] += 1

        t = (i+1)*buk_len+total_offset
        while t < len(dev) and dev[t][score_idx] == buk_info[i]['u_b']:
            buk_info[i]['total_cnt'] += 1
            t += 1
            total_offset += 1

        buk_info[i]['ratio'] = buk_info[i]['total_cnt'] / total_cnt
        buk_info[i]['e_idx'] = (i+1)*buk_len+total_offset-1

    #check bucket segmentations
    #print buk_info

    oot.sort(key=lambda x: x[score_idx])
    i = 0
    for j in range(len(oot)):
        while oot[j][score_idx] > buk_info[i]['u_b']:
            buk_info[i]['actual_ratio'] = buk_info[i]['actual_total'] / len(oot)
            i += 1
        buk_info[i]['actual_total'] += 1
    if len(oot) > 0:
        buk_info[i]['actual_ratio'] = buk_info[i]['actual_total'] / len(oot)
    #check bucket result
    #print buk_info
    x = []
    psi = 0.
    for i in range(buckets):
        if buk_info[i]['actual_ratio'] > 0 and buk_info[i]['ratio'] > 0:
            x.append(buk_info[i]['actual_ratio'])
            psi += (buk_info[i]['actual_ratio']-buk_info[i]['ratio']) * \
                   math.log(buk_info[i]['actual_ratio']/buk_info[i]['ratio'])
    return psi,x


dev = read_set(dev_fn)
oot = read_set(oot_fn)

psi,x = cal_psi(dev, oot, 0)

print 'For checking score\'s stability, psi = %.4f' % (psi)
