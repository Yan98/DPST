#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import re
from dataclasses import dataclass

PREFIX = None ### To be set

if PREFIX == None:
    raise SystemExit("Please set path prefix")

denovo_path = [
    ("cross.cat.mgf.test.repeat","high.bacillus.PXD004565"),
    ]

@dataclass
class Feature:
    spec_id: str
    mz: str
    z: str
    rt_mean: str
    seq: str
    scan: str

    def to_list(self):
        return [self.spec_id, self.mz, self.z, self.rt_mean, self.seq, self.scan, "0.0:1.0", "1.0"]

def transfer_mgf(old_mgf_file_name, output_feature_file_name):
    with open(old_mgf_file_name, 'r') as fr:
        with open(output_feature_file_name, 'w') as fw:
            writer = csv.writer(fw, delimiter=',')
            header = ["spec_group_id","m/z","z","rt_mean","seq","scans","profile","feature area"]
            writer.writerow(header)
            flag = False
            for line in fr:
                if "BEGIN ION" in line:
                    flag = True
                elif not flag:
                    pass
                elif line.startswith("TITLE="):
                    pass
                elif line.startswith("PEPMASS="):
                    mz = re.split("=|\r|\n", line)[1]
                elif line.startswith("CHARGE="):
                    z = re.split("=|\r|\n|\+", line)[1]
                elif line.startswith("SCANS="):
                    scan = re.split("=|\r|\n", line)[1]
                elif line.startswith("RTINSECONDS="):
                    rt_mean = re.split("=|\r|\n", line)[1]
                elif line.startswith("SEQ="):
                    seq = re.split("=|\r|\n", line)[1]
                elif line.startswith("END IONS"):
                    feature = Feature(spec_id=scan, mz=mz, z=z, rt_mean=rt_mean, seq=seq, scan=scan)
                    writer.writerow(feature.to_list())
                    flag = False
                    del scan
                    del mz
                    del z
                    del rt_mean
                    del seq
                       
                    