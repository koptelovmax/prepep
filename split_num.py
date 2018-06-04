import numpy as np
#%%
total_features = 192
f = open('phantoms_pains.sdf', 'r')
#file_freq = open("list_freq.txt", "w")
#file_infreq = open("list_infreq.txt", "w")
f_freq = open("freq.sdf", "w")
f_infreq = open("infreq.sdf", "w")
f_freq_num = open("num_freq.csv", "w")
f_infreq_num = open("num_infreq.csv", "w")
#count = 0
#count_start = 0
#count_end = 0
#molecule_id = 0
assay = np.zeros(6,bool)
buff_str = ''
for line in f:
    try:
        buff_str += line
        #if '<PUBCHEM_CID>' in line:
        #    line = f.next()
        #    molecule_id = np.int(line)
        #    buff_str += line
        if '<AID_623870>' in line:
            line = f.next()
            #count += 1
            assay[0] = np.int(line)
            buff_str += line                   
        elif '<AID_624168>' in line:
            line = f.next()
            #count += 1
            assay[1] = np.int(line)
            buff_str += line         
        elif '<AID_651704>' in line:
            line = f.next()
            #count += 1
            assay[2] = np.int(line)
            buff_str += line           
        elif '<AID_651724>' in line:
            line = f.next()
            #count += 1
            assay[3] = np.int(line)
            buff_str += line           
        elif '<AID_651725>' in line:
            line = f.next()
            #count += 1
            assay[4] = np.int(line)
            buff_str += line           
        elif '<AID_743445>' in line:
            line = f.next()
            #count += 1
            assay[5] = np.int(line)
            buff_str += line
        elif '<apol>' in line:
            num_features = np.zeros(total_features,float)
            i = 0
            line = f.next()
            buff_str += line
            num_features[i] = np.float(line)
            while i < total_features-1:
                i += 1
                line = f.next()
                buff_str += line
                line = f.next()
                buff_str += line
                line = f.next()
                buff_str += line
                num_features[i] = np.float(line)
            while not '$$$$' in line:
                line = f.next()
                #count += 1
                buff_str += line
            #count_end = count                  
            if np.sum(assay) > 1: #molecule is frequent
                #print molecule_id, assay
                #print count_start, count_end, count_end-count_start
                #file_freq.write(str(count_start)+' '+str(count_end)+'\n')
                f_freq.write(buff_str)
                num_buff_str = ''
                for el in num_features:
                    num_buff_str += str(el)+','
                f_freq_num.write(num_buff_str+'\n')
            else: #molecule is infrequent
                f_infreq.write(buff_str)
                num_buff_str = ''
                for el in num_features:
                    num_buff_str += str(el)+','
                f_infreq_num.write(num_buff_str+'\n')
                #print molecule_id, assay
                #print count_start, count_end, count_end-count_start
                #file_infreq.write(str(count_start)+' '+str(count_end)+'\n')  
                ######count_start = count+1
            buff_str = ''
        #count += 1
        #buff_str += line
    except ValueError:
        print "Invalid input:", line
f.close()
f_freq.close()
f_infreq.close()
f_freq_num.close()
f_infreq_num.close()
#%%
