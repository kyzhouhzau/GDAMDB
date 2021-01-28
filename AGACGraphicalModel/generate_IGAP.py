#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import pandas as pd
from collections import defaultdict
import sys
import re

def build_geneid_symbol(file="9606gene.txt"):
    hashdirsymbol = {}
    hashdirsynomy = {}
    with open(file) as rf:
        for line in rf:
            #TODO
            line = re.sub(', ',",",line)
            contents = line.strip().split()
            taxid = contents[0]
            Geneid = contents[3]
            symbol = contents[6]
            aliases = contents[7].split(",")
            if taxid=="9606":
                hashdirsymbol[Geneid] = symbol
                hashdirsynomy[Geneid] = tuple(aliases) 
    return hashdirsymbol,hashdirsynomy

def generate_read(file):
    id2gene,id2aliase = build_geneid_symbol()
    gene2id = {gene:id for id,gene in id2gene.items()}
    aliase2id = {value:key for key,value in id2aliase.items()}
    name_p_value = defaultdict(list)
    rsset = set()
    with open(file) as rf:
        rf.readline()
        for line in rf:
            contents = line.strip().split('\t')
            rs = contents[3]
            gene = contents[8]
            rsset.add(rs)
            geneid = gene2id.get(gene,-1)
            if geneid!=-1:
                name_p_value[geneid].append(float(contents[4]))
            else:
                # print("这些基因找不到ID:{}".format(gene))
                flag = False
                for key in aliase2id.keys():
                    if gene in key:
                        flag=True
                        geneid = aliase2id[key]
                        name_p_value[geneid].append(float(contents[4]))
                if not flag:
                    print("这些基因找不到ID:{}".format(gene))
    print("一共使用了{}个SNP".format(len(rsset)))
    #{"gene":p_value}
    return name_p_value
def gene_function(file):
    geneid2symbol,id2aliase = build_geneid_symbol()
    
    gene2function=defaultdict(set)
    with open(file) as rf:
        for line in rf:
            contens = line.strip().split()
            geneids = contens[0].split(';')
            for geneid in geneids:
                gene = geneid2symbol.get(geneid,-1)
                if gene==-1:
                    print("these gene can't find symbol:{}".format(geneid))
                    pass
                else:
                    function = contens[1]
                    if gene!="NA":
                        if function=="LOF":
                            function=0
                            gene2function[geneid].add(function)
                        elif function=="GOF":
                            function=1
                            gene2function[geneid].add(function)
                        elif function=="REG":
                            if geneid not in gene2function:
                                gene2function[geneid].add(function)
                        if function=="COM":
                            gene2function[geneid].add(0)
                            gene2function[geneid].add(1)             
    return gene2function


def build_data(gwasfile,name_p_value,gene2function,threlod,use_p_norm,norm_value):
    geneid2symbol,id2aliase = build_geneid_symbol()
    wf = open(gwasfile.split('.')[0]+".txt",'w')
    wf.write("GCST\t")
    xianzhucount=0
    deleted = 0
    loc=0
    total=0
    xianzhu = 0
    wxianzhu = open("三元组是否显著.txt",'w')
    wbuxianzhu = open("./generate/target_buxianzhu.txt",'w')
    gw = open("./generate/GWASxianzhu.txt",'w')
    for gene, ps in name_p_value.items():
        function = gene2function.get(gene,-1)
        # if function!=-1:
        #     if len(function)!=1:
        #         print("{}基因既有LOF又有GOF".format(gene))
        if function!=-1:
            functions=list(function)#[0,1]
        else:
            functions = [function]#[-1]
        p = min(ps)
        if float(p) < threlod:
            xianzhucount += 1
            line = "{}\t{}\n".format(geneid2symbol[gene],p)
            gw.write(line)
        if len(functions)==2:
            # function = 2
            function = 0
        else:
            function=functions[0]
            if function!=-1:
                function=0
        if gene in gene2function.keys():
            if p<threlod:
                genesymbol = geneid2symbol[gene]
                wxianzhu.write("{}\t{}\t{}\t{}\n".format(gene,function,genesymbol,p))
                xianzhu+=1
            else:
                wbuxianzhu.write("{}\t{}\t{}\n".format(gene,function,p))

        if p>threlod:
            function=-1

        if eval(use_p_norm):
            if p>threlod:
                p=eval(norm_value)
        token = "{}&{}&{}\t".format(gene,function,p)
        wf.write(token)
        total+=1
    if eval(use_p_norm):
        print("0.使用p值标准化策律并且非显著的p值被标准化为：{}".format(eval(norm_value)))
    else:
        print("0.不使用p值标准化策律")
    print("1.针对当前的GWAS和AGAC三元组数据，我们采用的阈值是{}".format(threlod))
    print("2.当前GWAS数据统计：\n---基因：{}\n---显著基因（p值小于Threshold）:{}".format(total,xianzhucount))
    print("3.当前AGAC三元组统计：\n---三元组(Gene;Function_change-0/1/2-LOF/GOF/COM;p-value) :{}".format(len(gene2function))) 
    print("---三元组（仅包含GWAS显著基因）:{}".format(xianzhu))

def countgenefunctionpair(gene2function):
    total = 0
    for value in gene2function.values():
        total += len(value)
    print("一共有{}个基因function对！".format(total))

def main():
    args = sys.argv
    if len(args)<=3:
        print("USAGE: python generate.py [threlod] [use_p_norm] [norm_value] [file_name]")
        print("USAGE note: [threshold]=[Arg1]/751851 \n USAGE note: [use_p_norm]=True, p-value greater than threshold are assigned with 0.5; [use_p_norm]=False, p-value unchange.")
        exit()
    threlod = args[1]
    use_p_norm = args[2]
    norm_value = args[3]
    file_=args[4]

    name_p_value = generate_read(file_)
    # read data from triples
    gene2function = gene_function("data/pubtator_Alzheimer'sdisease.txt")
    # count gene function pair
    countgenefunctionpair(gene2function)
    # generate data
    build_data(file_,name_p_value,gene2function,float(threlod),use_p_norm,norm_value)

if __name__=="__main__":
    main()


